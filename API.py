import datetime
from CNN import TextCNN
from tensorflow.contrib import learn
import DataExtractor
import numpy as np
from flask import Flask
import tensorflow as tf
import pickle

from flask import request,make_response

with tf.device('/cpu:0'):
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()



    TrainDatapath = "resources/traindata_10_02"
    # rel_id_Map = {}
    # rel_id_Map[0] = "REL$/business/company/founders"
    # rel_id_Map[1] = "REL$/people/person/nationality"
    # rel_id_Map[2] = "REL$/organization/parent/child"
    # rel_id_Map[3] = "REL$/location/neighborhood/neighborhood_of"
    # rel_id_Map[4] = "REL$/people/person/parents"

    file = open("resources/rel_id_map.pckl", 'rb')
    rel_id_Map = pickle.load(file)
    file.close()

    x_text_train, y_train = DataExtractor.load_data_and_labels_new(TrainDatapath)
    max_document_length = max([len(x.split(" ")) for x in x_text_train])

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
    learn.preprocessing.VocabularyProcessor.restore('runs/1486738068/vocab')
    print "Session restored"
    app = Flask(__name__)


    @app.route('/')
    def Start():
        return "working"


    @app.route('/GetPrediction/',methods = ['POST'])
    def TestpatternUtil():
        print "request received"

    x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))

    session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    # sess = tf.Session()
    cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )


    def Test_pattern(sess,x_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.dropout_keep_prob: 1.0
        }
        prediction = sess.run(
            cnn.predictions,
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # if writer:
            # writer.add_summary(summaries, step)
        return prediction


    global_step = tf.Variable(0, name="global_step", trainable=False)

    saver = tf.train.Saver()
    saver.restore(sess, "runs/1486738068/checkpoints/model-7800")



    print "Session restored"
    app = Flask(__name__)


    @app.route('/')
    def Start():
        return "working"


    @app.route('/GetPrediction/',methods = ['POST'])
    def TestpatternUtil():
        print "request received"
        sent = request.data
        list_sents = []
        list_sents.append(sent)
        x_test = np.array(list(vocab_processor.fit_transform(list_sents)))
        prediction = Test_pattern(sess,x_test)
        predicted_relation = rel_id_Map[prediction[0]]
        print "predicted relation:" + predicted_relation
        return predicted_relation

    if __name__ == '__main__':
        app.run(host='0.0.0.0',port=5002)