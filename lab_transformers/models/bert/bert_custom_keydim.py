import math
import warnings
from typing import Dict, Optional, Tuple, Union


import numpy as np
import tensorflow as tf
from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFMaskedLMOutput,
)
from transformers.modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    get_initializer,
    keras_serializable,
    unpack_inputs,
    TFPreTrainedModel,
)
from transformers.tf_utils import shape_list, stable_softmax
from transformers.utils import logging, DUMMY_INPUTS
from transformers import BertConfig

logger = logging.get_logger(__name__)


class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
            # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
            tf.debugging.assert_less(
                input_ids,
                tf.cast(self.vocab_size, dtype=input_ids.dtype),
                message=(
                    "input_ids must be smaller than the embedding layer's input dimension (got"
                    f" {tf.math.reduce_max(input_ids)} >= {self.vocab_size})"
                ),
            )
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(
                    start=past_key_values_length,
                    limit=input_shape[1] + past_key_values_length,
                ),
                axis=0,
            )

        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids
        )
        token_type_embeds = tf.gather(
            params=self.token_type_embeddings, indices=token_type_ids
        )
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


class TFBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size
        # self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(
            tensor=tensor,
            shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size),
        )

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(inputs=encoder_hidden_states), batch_size
            )
            value_layer = self.transpose_for_scores(
                self.value(inputs=encoder_hidden_states), batch_size
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size
            )
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size
            )
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size
            )
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size
            )

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFBertSelfAttention(config, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class TFBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFBertAttention(config, name="attention")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = TFBertAttention(config, name="crossattention")
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        self.bert_output = TFBertOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: Optional[tf.Tensor],
        encoder_attention_mask: Optional[tf.Tensor],
        past_key_value: Optional[Tuple[tf.Tensor]],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output,
            input_tensor=attention_output,
            training=training,
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [
            TFBertLayer(config, name=f"layer_._{i}")
            for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: Optional[tf.Tensor],
        encoder_attention_mask: Optional[tf.Tensor],
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]],
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if (
                    self.config.add_cross_attention
                    and encoder_hidden_states is not None
                ):
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output


class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states


class TFBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(
        self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.transform = TFBertPredictionHeadTransform(config, name="transform")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape: tf.TensorShape):
        self.bias = self.add_weight(
            shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias"
        )

        super().build(input_shape)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(
            a=hidden_states, b=self.input_embeddings.weight, transpose_b=True
        )
        hidden_states = tf.reshape(
            tensor=hidden_states, shape=[-1, seq_length, self.vocab_size]
        )
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


class TFBertMLMHead(tf.keras.layers.Layer):
    def __init__(
        self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs
    ):
        super().__init__(**kwargs)

        self.predictions = TFBertLMPredictionHead(
            config, input_embeddings, name="predictions"
        )

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores


@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        self.embeddings = TFBertEmbeddings(config, name="embeddings")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]

        if attention_mask is None:
            attention_mask = tf.fill(
                dims=(batch_size, seq_length + past_key_values_length), value=1
            )

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            training=training,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # Copied from `modeling_tf_t5.py`
        # Provided a padding mask of dimensions [batch_size, mask_seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(
                extended_attention_mask,
                (
                    attention_mask_shape[0],
                    1,
                    attention_mask_shape[1],
                    attention_mask_shape[2],
                ),
            )
            if past_key_values[0] is not None:
                # attention_mask needs to be sliced to the shape `[batch_size, 1, from_seq_length - cached_seq_length, to_seq_length]
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(
            extended_attention_mask, dtype=embedding_output.dtype
        )
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(
            tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst
        )

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(
                encoder_attention_mask, dtype=extended_attention_mask.dtype
            )
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(hidden_states=sequence_output)
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        dummy = {"input_ids": tf.constant(DUMMY_INPUTS)}
        # Add `encoder_hidden_states` to make the cross-attention layers' weights initialized
        if self.config.add_cross_attention:
            batch_size, seq_len = tf.constant(DUMMY_INPUTS).shape
            shape = (batch_size, seq_len) + (self.config.hidden_size,)
            h = tf.random.uniform(shape=shape)
            dummy["encoder_hidden_states"] = h

        return dummy


class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"cls.seq_relationship",
        r"cls.predictions.decoder.weight",
        r"nsp___cls",
    ]

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = TFBertMainLayer(config, add_pooling_layer=False, name="bert")
        self.mlm = TFBertMLMHead(
            config, input_embeddings=self.bert.embeddings, name="mlm___cls"
        )

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        warnings.warn(
            "The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.",
            FutureWarning,
        )
        return self.name + "/" + self.mlm.name + "/" + self.mlm.predictions.name

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        loss = (
            None
            if labels is None
            else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )

        return TFMaskedLMOutput(
            logits=output.logits, hidden_states=hs, attentions=attns
        )