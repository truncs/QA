from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from functools import reduce
from operator import mul
from os.path import join as pjoin

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
    flat_out = rnn_cell_impl._linear(flat_args, output_size, bias, 
                                     bias_initializer=tf.zeros_initializer())
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_wd(wd)

    return out


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        with tf.variable_scope('trans'):
            trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
            trans = tf.nn.relu(trans)
        with tf.variable_scope('gate'):
            gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out

def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur
    
# TODO do you need this?   
class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                       for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state
    
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    
    print(inputs.get_shape(), flat_inputs.get_shape(), 'Flattening')
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    print(flat_fw_outputs.get_shape(), fw_outputs.get_shape(), 'Reconstruct')
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out

def softsel(target, logits, mask=None, scope=None):
    """

    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input, is_train, hparams):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        context_embed, question_embed = inputs
        p_mask, q_mask = masks
        batch_size = hparams.batch_size,
        input_keep_prob = hparams.input_keep_prob

        cell = BasicLSTMCell(self.size, state_is_tuple=True)

        d_cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=input_keep_prob)

        p_len = tf.reduce_sum(tf.cast(p_mask, 'int32'), 1)  # [N]
        q_len = tf.reduce_sum(tf.cast(q_mask, 'int32'), 1)  # [N]


        with tf.variable_scope('prepro'):
            
            # [N, J, d], [N, d]
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(
                d_cell, d_cell, question_embed, q_len, dtype='float', scope='u1') 

            tf.get_variable_scope().reuse_variables()

            # [N, JX, 2d]
            (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, context_embed, 
                                                         p_len, dtype='float', scope='u1')  
            u = tf.concat([fw_u, bw_u], 2)
            h = tf.concat([fw_h, bw_h], 2)

        # Attention Layer
        with tf.variable_scope('attention_layer'):
            JQ = tf.shape(u)[1]
            JX = tf.shape(h)[1]

            h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, JQ, 1])
            u_aug = tf.tile(tf.expand_dims(u, 1), [1, JX, 1, 1])

            h_mask_aug = tf.tile(tf.expand_dims(p_mask, 2), [1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(q_mask, 1), [1, JX, 1])
            hu_mask = tf.cast(h_mask_aug, tf.bool) & tf.cast(u_mask_aug, tf.bool)
            hu_aug = h_aug * u_aug
            u_logits = linear_logits([h_aug, u_aug, hu_aug], True, scope='u_logits', 
                                     mask=hu_mask, is_train=is_train)

            u_a = softsel(u_aug, u_logits)  # [N, JX, d]
            h_a = softsel(h, tf.reduce_max(u_logits, 2))  # [N, d]
            h_a = tf.tile(tf.expand_dims(h_a, 1), [1, JX, 1])

            p0 = tf.concat([h, u_a, h * u_a, h * h_a], 2)
        return p0


class Decoder(object):
    def __init__(self, state_size):
        self.state_size = state_size

    def decode(self, knowledge_rep, masks, is_train, hparams):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        p0 = knowledge_rep
        p_mask, q_mask = masks
        batch_size = hparams.batch_size
        input_keep_prob = hparams.input_keep_prob

        p_len = tf.reduce_sum(tf.cast(p_mask, 'int32'), 1)  # [N]
        q_len = tf.reduce_sum(tf.cast(q_mask, 'int32'), 1)  # [N]

        JX = tf.shape(p_mask)[1]

        with tf.variable_scope("main"):
            cell = BasicLSTMCell(self.state_size, state_is_tuple=True)
            first_cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=input_keep_prob)

            # [N, JX, 2d]
            (fw_g0, bw_g0), _ = _bidirectional_dynamic_rnn(first_cell, first_cell, p0, 
                                                           p_len, dtype='float', scope='g0') 
            g0 = tf.concat([fw_g0, bw_g0], 2)
            
            cell = BasicLSTMCell(self.state_size, state_is_tuple=True)
            first_cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=input_keep_prob)

            # [N, JX, 2d]
            (fw_g1, bw_g1), _ = _bidirectional_dynamic_rnn(first_cell, first_cell, g0, 
                                                           p_len, dtype='float', scope='g1')  
            g1 = tf.concat([fw_g1, bw_g1], 2)
            logits = linear_logits([g1, p0], self.state_size, 0.0, scope='logits1', 
                                   mask=p_mask, is_train=is_train)

            # TODO use batch _size
            a1i = softsel(tf.reshape(g1, [batch_size, JX, 2 * self.state_size]), 
                          tf.reshape(logits, [batch_size, JX]))

            a1i = tf.tile(tf.expand_dims(a1i, 1), [1, JX, 1])
            
            flat_logits1 = tf.reshape(logits, [-1, JX])
            flat_yp = tf.nn.softmax(flat_logits1)  # [-1, M*JX]
            yp1 = tf.reshape(flat_yp, [-1, JX])

            cell = BasicLSTMCell(self.state_size, state_is_tuple=True)
            d_cell = SwitchableDropoutWrapper(cell, is_train, input_keep_prob=input_keep_prob)

            # [N, M, JX, 2d]
            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell, d_cell, 
                                                          tf.concat([p0, g1, a1i, g1 * a1i], 2),
                                                          p_len, dtype='float', scope='g2') 
            g2 = tf.concat([fw_g2, bw_g2], 2)
            logits2 = linear_logits([g2, p0], self.state_size, 0.0, scope='logits2', 
                                    mask=p_mask, is_train=is_train)
            flat_logits2 = tf.reshape(logits2, [-1, JX])
            flat_yp = tf.nn.softmax(flat_logits2)  # [-1, M*JX]
            yp2 = tf.reshape(flat_yp, [-1, JX])
        return (yp1, flat_logits1), (yp2, flat_logits2)


class QASystem(object):
    def __init__(self, encoder, decoder, hparams):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.encoder = encoder
        self.decoder = decoder
        self.config = hparams

        # Placeholders
        self.np_embeddings = tf.placeholder(
            'float', [self.config.vocab_size, self.config.embedding_size], name='np_embeddings')

        self.is_train = tf.placeholder(tf.bool, shape=())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embeddings_init = self.setup_embeddings()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self, p, q, p_mask, q_mask, span):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        # Initialize the various variables
        self.p = p
        self.q = q
        self.p_mask = p_mask
        self.q_mask = q_mask
        self.span = span

        # Lookup the embedding for context and question
        p = tf.nn.embedding_lookup(self.embeddings, self.p)
        q = tf.nn.embedding_lookup(self.embeddings, self.q)

        # TODO make wd configurable
        with tf.variable_scope('highway'):
            p = highway_network(p, 2, True, wd=0.0, is_train=self.is_train)
            tf.get_variable_scope().reuse_variables()
            q = highway_network(q, 2, True, wd=0.0, is_train=self.is_train)
    
        knowledge_rep = self.encoder.encode((p, q), (self.p_mask, self.q_mask), None, 
                                            self.is_train, self.config)
        
        # TODO doesn't have JX and the masks
        (yp1, flat_logits1), (yp2, flat_logits2) = self.decoder.decode(knowledge_rep, 
                                                        (self.p_mask, self.q_mask), 
                                                        self.is_train,  self.config)
        self.yp1 = yp1
        self.yp2 = yp2
        self.flat_logits1 = flat_logits1
        self.flat_logits2 = flat_logits2


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            span1, span2 = tf.split(self.span, num_or_size_splits=2, axis=1)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(span1, [-1]), logits=self.flat_logits1, name='loss1')

            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(span2, [-1]), logits=self.flat_logits2, name='loss2')

            loss = tf.reduce_mean(
                tf.reshape(loss1, [-1, 1]) * tf.to_float(self.p_mask)) + tf.reduce_mean(tf.reshape(loss2, [-1, 1]) * tf.to_float(self.p_mask))

            self.loss = loss

            
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """

        with vs.variable_scope("embeddings"):

            self.embeddings = tf.Variable(tf.constant(0.0, 
                                    shape=[self.config.vocab_size, self.config.embedding_size]),
                                    trainable=False, name="embeddings")
            embedding_init = self.embeddings.assign(self.np_embeddings)
            return embedding_init
            

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = train_x

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = train_y

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        #input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.yp1, self.yp2]

        outputs = session.run(output_feed, test_x)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def make_dataset_iterator(self, context_file_path, question_file_path, span_file_path,
                           batch_size=10, epoch_size=2):
        """
        Given file paths of context, question and span this function creates a one-shot
        iterator.

        :param context_file_path: File path of context data.
        :param question_file_path: File path of question data.
        :param span_file_path: File path of context data.
        :param batch_size: Size of the required batch.
        :param epoch_size: Size of the required number of epochs.
        :return: Dataset iterator with the following shape
        ((context_vec, context_mask), (question_vec, question_mask), (span_vec))
        """
        def build_mask(x):
            return tf.ones_like(x, dtype=tf.int32)

        def make_dataset(filenames, with_mask=False):
            dataset = tf.contrib.data.TextLineDataset(filenames)

            dataset = (dataset.map(lambda line: tf.string_split([line]).values)
                    .map(lambda strings: tf.string_to_number(strings, out_type=tf.int32)))
            if with_mask:
                dataset = dataset.map(lambda x: (x, build_mask(x)))
            return dataset
        
        dataset1 = make_dataset([context_file_path], with_mask=True)
        dataset2 = make_dataset([question_file_path], with_mask=True)
        dataset3 = make_dataset([span_file_path])

        dataset = tf.contrib.data.Dataset.zip((dataset1, dataset2, dataset3))
        dataset = (dataset.padded_batch(self.config.batch_size, 
                    padded_shapes=(([None], [None]), ([None], [None]), ([None])))
                    .repeat(self.config.epochs)
                    .shuffle(buffer_size=self.config.dataset_buffer_size))
        iterator = dataset.make_one_shot_iterator()
    
        return iterator

    def train(self, session, dataset, train_dir, embeddings):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        
        context_file_path, question_file_path, span_file_path = dataset

        iterator = self.make_dataset_iterator(context_file_path, question_file_path, 
                                              span_file_path, self.config.batch_size, 
                                              self.config.epochs)

        # Global Step
        global_step = tf.Variable(0, trainable=False, name='global_step')

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step,
                                           20000, 0.96, staircase=True)

        opt_fn = get_optimizer(self.config.optimizer)

        # Load embeddings
        session.run(self.embeddings_init, feed_dict={self.np_embeddings: embeddings})

        # Setup the model and loss
        context_tuple, question_tuple, span = iterator.get_next()
        p, p_mask = context_tuple
        q, q_mask = question_tuple
        span = span

        self.setup_system(p, q, p_mask, q_mask, span)
        self.setup_loss()

        learning_step = (
            opt_fn(learning_rate)
            .minimize(self.loss)
        )

        # Initialize all variables
        session.run(tf.global_variables_initializer())

        # Model saver
        saver = tf.train.Saver()

        while True:
            try:
                
                feed_dict = {
                    self.is_train: True
                }

                loss, step, _ = self.optimize(session, feed_dict, 
                                           [self.loss, global_step, learning_step])

                print('Loss value of %.2f at step %d' % (loss, step))

                if (step + 1) % 500 == 0:
                    save_path = saver.save(session, '%s/model%d.ckpt' % (train_dir, step))
                    print("Model saved in file: %s" % save_path)
                    
                # TODO: Implement gradient clipping over here
                global_step = global_step + 1

            except tf.errors.OutOfRangeError:
                break
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
