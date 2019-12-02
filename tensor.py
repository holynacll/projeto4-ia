import os
import numpy as np
import tensorflow as tf

#r = np.genfromtxt(os.getcwd()+'/database/treino/comedia', delimiter=',', dtype={'names': ('titulo', 'tipo', 'sinopse'), 'formats': ('S20', 'S10', 'S200')})
#print(r[0])
#arr = np.char.split(r[0], sep=',')
#print(arr, arr.shape)

f = open(os.getcwd()+'/database/treino/comedia', 'r')

listaTituloTreino = np.array([])
listaTipoTreino =  np.array([])
listaSinopseTreino =  np.array([])
for line in f:
    lista = np.array(line.split(','))
    listaTituloTreino = np.append(listaTituloTreino, lista[0])
    listaTipoTreino =  np.append(listaTipoTreino, lista[1])
    listaSinopseTreino =  np.append(listaSinopseTreino, lista[2])

f.close()

f = open(os.getcwd()+'/database/treino/drama', 'r')

for line in f:
    lista = np.array(line.split(','))
    listaTituloTreino = np.append(listaTituloTreino, lista[0])
    listaTipoTreino =  np.append(listaTipoTreino, lista[1])
    listaSinopseTreino =  np.append(listaSinopseTreino, lista[2])

f.close()

f = open(os.getcwd()+'/database/teste/comedia', 'r')

listaTituloTeste = np.array([])
listaTipoTeste =  np.array([])
listaSinopseTeste =  np.array([])
for line in f:
    lista = np.array(line.split(','))
    listaTituloTeste = np.append(listaTituloTeste, lista[0])
    listaTipoTeste =  np.append(listaTipoTeste, lista[1])
    listaSinopseTeste =  np.append(listaSinopseTeste, lista[2])

f.close()

f = open(os.getcwd()+'/database/teste/drama', 'r')

for line in f:
    lista = np.array(line.split(','))
    listaTituloTeste = np.append(listaTituloTeste, lista[0])
    listaTipoTeste =  np.append(listaTipoTeste, lista[1])
    listaSinopseTeste =  np.append(listaSinopseTeste, lista[2])

f.close()

X_train = listaSinopseTreino
X_test = listaSinopseTeste
y_train = listaTipoTreino
y_test = listaTipoTeste


vocabulary = {}
max_vocab = 5000

for sentence in X_train:
    sentence = sentence.split(' ')
    for word in sentence:
        if word in vocabulary and word != ' ' and word != '\n' and word != '':
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1

 
#print(len(vocabulary), max(vocabulary, key=vocabulary.get))
word_to_int = {}
for counter, word in enumerate(sorted(vocabulary, key=vocabulary.get, reverse=True)):
    #print(counter, word, vocabulary[word])
    word_to_int[word] = counter
    if(counter == max_vocab-1):
        break


# X_train vetorizando a entrada
X_train2 = np.zeros([len(listaSinopseTreino), max_vocab] ,dtype = np.int64)
X_test2 = np.zeros([len(listaSinopseTreino), max_vocab] ,dtype = np.int64)

for num_sentence, sentence in enumerate(X_train):
    sentence = sentence.split(' ')
    for word in sentence:
        if word in word_to_int and word != ' ' and word != '\n' and word != '':
            X_train2[num_sentence][word_to_int[word]] = 1

for num_sentence, sentence in enumerate(X_test):
    sentence = sentence.split(' ')
    for word in sentence:
        if word in word_to_int and word != ' ' and word != '\n' and word != '':
            X_test2[num_sentence][word_to_int[word]] = 1


# y_train one hot
y_train2 = np.empty([len(listaTipoTreino)])
y_test2 = np.empty([len(listaTipoTreino)])

for i, y in enumerate(y_train):
    if(y == 'comedia'):
        y_train2[i] = 0
    elif(y == 'drama'):
        y_train2[i] = 1

for i, y in enumerate(y_test):
    if(y == 'comedia'):
        y_test2[i] = 0
    elif(y == 'drama'):
        y_test2[i] = 1



graph = tf.Graph()
with graph.as_default():

    x = tf.compat.v1.placeholder(tf.float64, shape=(None, max_vocab))
    y = tf.compat.v1.placeholder(tf.int64, shape=(None,))
    lr = tf.compat.v1.placeholder(tf.float32)
    is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")


    #cl1 = tf.layers.conv1d(x, 32, (3), (1), padding="same", activation=tf.nn.relu, name = 'cl1')
    #mp1 = tf.layers.max_pooling2d(cl1, (2, 2), (2, 2), padding='same', name = 'mp1')


    #x_vector = tf.compat.v1.reshape(mp1, [-1, drop1.shape[1]*drop1.shape[2]*drop1.shape[3]])
    fc1 = tf.layers.dense(x, 64, activation=tf.nn.relu, name="fc1")
		
    drop = tf.layers.dropout(fc1, 40)

    fc2 = tf.layers.dense(drop, 32, activation=tf.nn.relu, name="fc2")
    fc3 = tf.layers.dense(fc2, 16, activation=tf.nn.relu, name="fc3")

    fc4 = tf.layers.dense(fc3, 4, name="fc4")

    x_norm = tf.layers.batch_normalization(x, training=is_train)

    y_one_hot = tf.one_hot(y, 2)

    loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits = fc4)

    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    train_op = tf.group([train_op, update_ops])

    correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(fc4, axis=1), y), dtype=tf.float32))


def accu(session, xi, yi):
    batch_size = 32
    cont = 0
    for i in range(0, len(xi), batch_size):
        x_batch = xi[i:i+batch_size]
        y_batch = yi[i:i+batch_size]
        ret = session.run([correct], feed_dict = {x : x_batch, y : y_batch, is_train : False})
        cont += ret[0]
    return 100.0*cont/len(xi)

with tf.compat.v1.Session(graph = graph) as session:
    session.run(tf.compat.v1.global_variables_initializer())

    learning_rate = 0.0001
    batch_size = 32
    for i in range(5000):
        idx = np.random.permutation(len(X_train2))[:batch_size]
        x_batch = np.take(X_train2, idx, axis = 0)
        y_batch = np.take(y_train2, idx, axis = 0)       
        ret = session.run([train_op], feed_dict = {x : x_batch, y : y_batch, lr : learning_rate, is_train : True})   
        if(i%100 == 99):
            print("Iteration #%d" % (i))
            print("TRAIN: ACC=%.5f" % (accu(session, X_train2, y_train2)))
            print("VAL: ACC=%.5f" % (accu(session, X_test2, y_test2)))


