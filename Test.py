import datetime
import tensorflow as tf
import numpy as np
from CNN import TextCNN
from tensorflow.contrib import learn
import DataExtractor
import pickle

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

    run_path = "runs/1487080266"
    file = open("resources/rel_id_map.pckl", 'rb')
    rel_id_Map = pickle.load(file)
    file.close()
    TrainDatapath = "resources/traindata_10_02"
    # rel_id_Map = {}
    # rel_id_Map[0] = "REL$/business/company/founders"
    # rel_id_Map[1] = "REL$/people/person/nationality"
    # rel_id_Map[2] = "REL$/organization/parent/child"
    # rel_id_Map[3] = "REL$/location/neighborhood/neighborhood_of"
    # rel_id_Map[4] = "REL$/people/person/parents"

    x_text_train, y_train = DataExtractor.load_data_and_labels_new(TrainDatapath)
    max_document_length = max([len(x.split(" ")) for x in x_text_train])

    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(run_path+'/vocab')

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
        prediction, CNN_output = sess.run(
            [cnn.predictions, cnn.h_pool_flat],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # if writer:
            # writer.add_summary(summaries, step)
        return prediction, CNN_output


    global_step = tf.Variable(0, name="global_step", trainable=False)

    saver = tf.train.Saver()
    saver.restore(sess, run_path+"/checkpoints/model-7800")
    count = 0
    lines = open('resources/traindata_10_02').readlines()
    fw = open('test_results/results_10_02','w')
    fw1 = open('test_results/hidden_states_results_10_02','w')

    for line in lines:
        splt = line.split("\t")
        pattern = splt[0]
        actual_relation = rel_id_Map[int(splt[1].strip("\n"))]
        list_sents = []
        list_sents.append(pattern)
        x_test = np.array(list(vocab_processor.fit_transform(list_sents)))
        prediction, cnn_output = Test_pattern(sess,x_test)
        predicted_relation = rel_id_Map[prediction[0]]
        if predicted_relation == actual_relation:
            count += 1
        fw.write(pattern + "\t" + actual_relation +  "\t" + predicted_relation + "\n")
        fw1.write(pattern+"\t"+actual_relation+"\t"+str(cnn_output).replace('\n',' ')+"\n")
        # result.append(predicted_relation)
    print "Correct predictions:" + str(count)
    fw.close()