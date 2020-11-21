from keras.models import Sequential
from keras.layers.core import Activation,RepeatVector,Dropout,Dense
from keras.layers import TimeDistributed,Input
from keras.models import Model
from keras.layers.recurrent import LSTM
import numpy as np 

# inputs = Input([STEP_SIZE,INPUT_SIZE])
# x = LSTM(CELL_SIZE,input_shape=(STEP_SIZE,INPUT_SIZE))(inputs)
# x = Dropout(0.25)(x)



'''
x = np.zeros((batch_size,seq_len,max_no),dtype=np.float32)  # x = (64,10,100)
#print(x)
y = np.zeros((batch_size,seq_len,max_no),dtype=np.float32)
X = np.random.randint(max_no,size=(batch_size,seq_len))     # X =(64,10)
Y = np.sort(X,axis=1)                                       #对X中的每一组从大到小进行排序

for ind, batch in enumerate(X):         #ind 0~63
    #print(batch)
    for j, elem in enumerate(batch):    #j 0~9
        x[ind, j, elem] = 1


for ind, batch in enumerate(Y):
    for j, elem in enumerate(batch):
        y[ind, j, elem] = 1


#batch_gen(batch_size,STEP_SIZE,INPUT_SIZE)
'''
#----------------------------------------------------------LSTM模型
batch_size = 64     #一次训练的数目
STEP_SIZE = 3
INPUT_SIZE = 4      #代表输入数字的范围
CELL_SIZE = 10      #输出的维度是10维

inputs = Input([STEP_SIZE,INPUT_SIZE])
x = LSTM(CELL_SIZE,input_shape=(STEP_SIZE,INPUT_SIZE))(inputs)
x = Dropout(0.25)(x)

# inputs = Input([STEP_SIZE,CELL_SIZE])
# x = LSTM(CELL_SIZE,input_shape=(STEP_SIZE,INPUT_SIZE))(inputs)
# x = Dropout(0.25)(x)

# CELL_SIZE->STEP_SIZE,CELL_SIZE
# 分为STEP_SIZE个时间序列
x = RepeatVector(STEP_SIZE)(x)

# STEP_SIZE,CELL_SIZE -> STEP_SIZE,NERVE_NUM
# 当return_sequences = True 时，会输出时间序列
# STEP_SIZE代表时间序列，CELL_SIZE代表每一个时间序列的输出
x = LSTM(CELL_SIZE,return_sequences=True)(x)

# 对每一个STEP进行全连接
x = TimeDistributed(Dense(INPUT_SIZE))(x)
x = Dropout(0.5)(x)
x = Activation('softmax')(x)

model = Model(inputs,x)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#-------------------------------------------------------------------------------展示
def encode(X,seq_len,vocab_size):
    x = np.zeros(shape=(len(X),seq_len,vocab_size),dtype=np.float32)
    for ind,batch in enumerate(X):
        for j,elem in enumerate(batch):
            x[ind,j,elem] = 1
    return x

def batch_gen(batch_size=32, seq_len=10, max_no=100):
    while(True):
        x = np.zeros((batch_size,seq_len,max_no),dtype=np.float32)
        y = np.zeros((batch_size,seq_len,max_no),dtype=np.float32)
        X = np.random.randint(max_no,size=(batch_size,seq_len))
        Y = np.sort(X,axis=1)

        for ind, batch in enumerate(X):
            for j, elem in enumerate(batch):
                x[ind, j, elem] = 1

        for ind, batch in enumerate(Y):
            for j, elem in enumerate(batch):
                y[ind, j, elem] = 1
            
        yield x,y

for ind, (X, Y) in enumerate(batch_gen(batch_size, STEP_SIZE, INPUT_SIZE)):
    loss, acc = model.train_on_batch(X, Y)
    if ind % 250 == 0:
        print("ind:",ind)
        testX = np.random.randint(INPUT_SIZE, size=(1, STEP_SIZE))
        test = encode(testX, STEP_SIZE, INPUT_SIZE)
        print("before is")
        print(testX)
        y = model.predict(test, batch_size=1)
        print("actual sorted output is")
        print(np.sort(testX))
        print("sorting done by LSTM is")
        print(np.argmax(y, axis=2))
        print("\n")



