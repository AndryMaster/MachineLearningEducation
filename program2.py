import tensorflow._api.v2.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = (1 - alpha) * prev_avg + alpha * curr_value

avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incoming_values", curr_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./tflogs")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # sess.add_graph(sess.graph)
    for i, value in enumerate(raw_data):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: value})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(f"Val: {value}\tAvg: {curr_avg}")
        writer.add_summary(summary_str, i)

writer.close()


# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# print(tf.config.list_physical_devices('GPU'))
