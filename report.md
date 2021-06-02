
# Giới thiệu về các thư viện phổ biến trong Python

###  I. OpenCV

OpenCV là dự án bắt đầu tại hãng Intel vào năm 1999 bởi Gary Bradsky và ra mắt lần đầu tiên vào năm 2000. Sau đó Vadim Pisarevsky gia nhập và quản lý nhóm. Vào năm 2005, OpenCV được sử dụng trên xe tự lái Stanley và chiếc này đã vô địch giải đấu 2005 DARPA Grand. Tiếp theo nó tiếp tục được cải tiến và phát triển dưới sự hỗ trợ của Willow Garage bên cạnh với Gary Bradsky và Vadim Pisarevsky. Hiện tại OpenCV là một thư viện mã nguồn mở hàng đầu cho thị giác máy tính (computer vision), xử lý ảnh và máy học, và các tính năng tăng tốc GPU trong hoạt động thời gian thực.

**_Các thao tác cơ bản_**

_a,Hiển thị hình ảnh_
``` python
**import** cv2  
  
img = cv2.imread(**'flower.jpg'**)  
cv2.imshow(**'Display Image'**, img)  
cv2.waitKey(0)
```
![](https://github.com/ductai202/Image/blob/main/1.png?raw=true)

Ở đây flower.jpg là file hình ảnh để test, hàm `_waitKey(0)_` là hàm chờ không cho thoát cửa sổ lập tức mà phải người dùng nhấn phím bất kỳ để thoát.

_b,Lấy kích thước ảnh_
```python
**import** cv2  
  
img = cv2.imread(**'flower.jpg'**)  
cv2.imshow(**'Display Image'**, img)  
cv2.waitKey(0) 
``` 
![](https://github.com/ductai202/Image/blob/main/2.png?raw=true)

_c,Cắt ảnh_
```python
img = cv2.imread(**'flower.jpg'**)  
roi = img[50:350, 60:360]  
cv2.imshow(**'Region Of Interest'**, roi)  
cv2.waitKey(0)
```
![](https://github.com/ductai202/Image/blob/main/3.png?raw=true)

_d,Thay đổi kích thước ảnh_
```python
img = cv2.imread(**'flower.jpg'**)  
(h, w, d) = img.shape  
r = 300.0 / w  
dim = (300, int(h * r))  
resized = cv2.resize(img, dim)  
cv2.imshow(**'Resize'**, resized)  
cv2.waitKey(0)
```
![](https://github.com/ductai202/Image/blob/main/4.png?raw=true)

_e,Vẽ đường_

Trong tất cả các hàm vẽ hình ,ta sẽ thấy một số đối số phổ biến như được đưa ra dưới đây:

-   _Img:_ Hình ảnh bạn muốn vẽ hình.
-   _Color:_ Màu sắc của hình dạng cho BGR, chuyển nó thành một bộ tuple, ví dụ: (255,0,0) cho màu xanh lam. Đối với thang độ xám, chỉ cần vượt qua giá trị vô hướng của nó.
-   _Độ dày (thickness):_ Độ dày của đường thẳng hoặc đường tròn, vv Nếu -1 được chuyển cho các hình dạng khép kín như hình tròn, nó sẽ lấp đầy hình dạng. độ dày mặc định = 1.
-   _LineType:_ Loại dòng, cho dù 8-kết nối, dòng chống răng cưa vv Theo mặc định, nó là 8-kết nối. cv2.LINE_AA cung cấp dòng chống răng cưa trông rất tuyệt vời cho các đường cong.
```python
**import** cv2  
**import** numpy **as** np  
  
_# Create a black image  
_img = np.zeros((512, 512, 3), np.uint8)  
_# Draw a diagonal blue line with thickness of 5 px  
_img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)  
cv2.imshow(**'Line'**, img)  
cv2.waitKey(0)  
```
![](https://github.com/ductai202/Image/blob/main/5.png?raw=true)

_f,Thêm văn bản vào hình ảnh_

Để đưa văn bản vào hình ảnh, cần chỉ định những điều sau đây:

-   Dữ liệu văn bản muốn viết.
-   Tọa độ vị trí của nơi muốn đặt nó (ví dụ: góc dưới cùng bên trái nơi dữ liệu bắt đầu).
-   Kiểu phông chữ (Kiểm tra tài liệu cv2.putText () cho phông chữ được hỗ trợ).
-   Quy mô phông chữ (chỉ định kích thước phông chữ).
-   Những thứ thông thường như màu sắc, độ dày, loại đường ... (Để có giao diện tốt hơn, sử dụng lineType = cv2.LINE_AA).
```python
**import** cv2  
**import** numpy **as** np  
  
_# Create a black image  
_img = np.zeros((512, 512, 3), np.uint8)  
font = cv2.FONT_HERSHEY_SIMPLEX  
cv2.putText(img, **'OpenCV'**, (10, 300), font, 4, (255, 255, 255), 2, cv2, LINE_AA)  
cv2.imshow(**'Text'**, img)  
cv2.waitKey(0)
```
![](https://github.com/ductai202/Image/blob/main/6.png?raw=true)

_g, Chuyển đổi hệ màu_
```python
**import** cv2  
  
img = cv2.imread(**'flower.jpg'**)  
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
cv2.imshow(**'Color'**, img)  
cv2.waitKey(0)  
```
![](https://github.com/ductai202/Image/blob/main/7.png?raw=true)

###  II. Thư viện Keras

**_Giới thiệu_**

Bản chất của bài toán Deep learning: Bạn có dữ liệu, bạn muốn máy tính học được các mô hình (model) từ dữ liệu, sau đó dùng mô hình đấy để dự đoán được các dữ liệu mới. Các bước cơ bản làm một bài toán Deep learning :

-   Xây dựng bài toán.
-   Chuẩn bị dữ liệu (dataset).
-   Xây dựng model.
-   Định nghĩa loss function.
-   Thực hiện backpropagation và áp dụng gradient descent để tìm các parameter gồm weight và bias để tối ưu loss function.
-   Dự đoán dữ liệu mới bằng model với các hệ số tìm được ở trên.

Keras là một framework mã nguồn mở cho Deep learning được viết bằng Python. Nó có thể chạy trên nền của các deep learning framework khác như: tensorflow, theano, CNTK. Với các API bậc cao, dễ sử dụng, dễ mở rộng, keras giúp người dùng xây dựng các deep learning model một cách đơn giản.

Mô hình chung bài toán CNN: _Input image -> Convolutional layer (Conv) + Pooling layer (Pool) -> Fully connected layer (FC) -> Output._

![](https://github.com/ductai202/Image/blob/main/8.png?raw=true)

Model CNN

**_Softmax function_**

Softmax là hàm kích hoạt ở lớp output trong một mạng Neural network được sử dụng với bài toán phân loại nhị phân với nhiều class (multi-class classification problems) ở lớp output.

Nhắc lại phần neural network, ở mỗi layer sẽ thực hiện 2 bước: tính tổng linear các node ở layer trước và thực hiện activation function (ví dụ sigmoid function, softmax function). Do sau bước tính tổng linear cho ra các giá trị thực

nên cần dùng **softmax function** dùng để chuyển đổi giá trị thực trong các node ở output layer sang giá trị phần trăm.

![](https://github.com/ductai202/Image/blob/main/9.png?raw=true)


**Với các bài toán classification (phân loại) thì nếu có 2 lớp thì hàm activation ở output layer là hàm sigmoid, còn nhiều hơn 2 lớp thì hàm activation ở ouput layer là hàm softmax**

**_Models_**

Trong Keras có hỗ trợ 2 cách dựng models là Sequential model và Function API. Với Sequential ta sử dụng như sau:
```python
from keras.models import Sequential  
from keras.layers import Dense, MaxPooling2D, Flatten, Convolution2D  
  
model = Sequential()  
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Flatten())  
model.add(Dense(output_dim=128, activation='relu'))  
model.add(Dense(output_dim=1, activation='sigmoid'))  
  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
  
model.fit(x_train, y_train,  
batch_size=batch_size,  
epochs=eposhs,  
verbose=1,  
validation_data=(x_test, y_test))
```
Nội dung đoạn code trên như sau:

1) Khởi tạo models Sequential ( ) để nói cho keras là ta sẽ xếp các layer lên nhau để tạo model.

2) Tạo Convolutionnal Layers : Conv2D là convolution dùng để lấy feature từ ảnh với các tham số :

-   _filters_ : số filter của convolution
-   _kernel_size :_ kích thước window search trên ảnh
-   _strides :_ số bước nhảy trên ảnh
-   _activation :_ chọn activation như linear, softmax, relu, tanh, sigmoid. Đặc điểm mỗi hàm các bạn có thể search thêm để biết cụ thể nó ntn.
-   _padding :_ có thể là "valid" hoặc "same". Với same thì có nghĩa là padding =1.
-   _data_format__:_ format channel ở đầu hay cuối.

3) Pooling Layers: sử dụng để làm giảm param khi train, nhưng vẫn giữ được đặc trưng của ảnh.

-   _pool_size :_ kích thước ma trận để lấy max hay average
-   _Ngoài ra còn có :_ MaxPooling2D, AvergaPooling1D, 2D ( lấy max , trung bình) với từng size.

4) Dense ( ): Layer này cũng như một layer neural network bình thường, với các tham số cần quan tâm:

-   _units :_ số chiều output, như số class sau khi train ( chó , mèo, lợn, gà).
-   _activation :_ chọn activation đơn giản với sigmoid thì output có 1 class.
-   _use_bias :_ có sử dụng bias hay không (True or False)
-   _kernel_initializer:_ khởi tạo giá trị đầu cho tham số trong layer trừ bias
-   _bias_initializer:_ khởi tạo giá trị đầu cho bias
-   _kernel_regularizer:_ regularizer cho coeff
-   _bias_regularizer:_ regularizer cho bias
-   _activity_regularizer:_ có sử dụng regularizer cho output ko
-   _kernel_constraint,bias_constraint:_ có ràng buộc về weight ko

5) Hàm compile: Ở hàm này chúng ta sử dụng để training models như thuật toán train qua optimizer như Adam, SGD, RMSprop,..

-   _learning_rate :_ dạng float , tốc độc học, chọn phù hợp để hàm số hội tụ nhanh.

6) Hàm fit ():

-   Bao gồm data train, test đưa vào training.
-   Batch_size thể hiện số lượng mẫu mà Mini-batch GD sử dụng cho mỗi lần cập nhật trọng số .
-   Epoch là số lần duyệt qua hết số lượng mẫu trong tập huấn luyện.
-   Giả sử ta có tập huấn luyện gồm 55.000 hình ảnh chọn batch-size là 55 images có nghĩa là mỗi lần cập nhật trọng số, ta dùng 55 images. Lúc đó ta mất 55.000/55 = 1000 iterations (số lần lặp) để duyệt qua hết tập huấn luyện (hoàn thành 1 epochs). Có nghĩa là khi dữ liệu quá lớn, chúng ta không thể đưa cả tập data vào train được, ta phải chia nhỏ data ra thành nhiều batch nhỏ hơn.
-   Ngoài ra ta có thể khai báo dùng function API như sau :
```python
from keras.models import Model  
from keras.layers import Input, Dense  
  
input = Input(shape=(64,))  
output = Dense(32)(input)  
model = Model(input=input, output=output)
```
Trong Keras có đưa ra các funtion xử lý các loại data như:

1. Sequence Preprocessing : tiền xử lý chuỗi.

-   _TimeseriesGenerrator :_ tạo data cho time series

-   _pad_sequences :_ padding các chuỗi có độ dài bằng nhau
-   _skipgrams :_ tạo data trong models skip gram, trả về 2 tuple nếu từ xuất hiện cùng nhau, là 1 nếu không có.

2. Text Preprocessing : tiền xử lý text

-   _Tokenizer :_ tạo token từ documment
-   _one_hot :_ tạo data dạng one hot encoding
-   _text_to_word_seqence :_ convert text thành sequence ..

3. Image Preprocessing : tiền xử lý image

-   _ImageDataGenerator :_ tạo thêm data bằng cách scale, rotation ,...để thêm data train.

**_Notes_**

## +) **Loss_funtion**

-   _mean_squared_eror:_ thường dùng trong regression tính theo eculid.
-   _mean_absolute_error :_ để tính giá trị tuyệt đối.
-   _binary_crossentropy :_ dùng cho classifier 2 class.
-   _categorical_crossentropy :_ dùng classifier nhiều class.

## +) **Metrics**

Để đánh giá accuracy của models.

-   _binary_accuracy :_ dùng cho 2 class , nếu y_true== y_predict thì trả về 1 ngược lại là 0.
-   _categorical_accuracy :_ cũng giống như trên nhưng cho nhiều class.

## +) **Optimizers**

Dùng để chọn thuật toán training.

-   _SGD:_  Stochastic Gradient Descent optimizer.
-   _RMSprop_:  RMSProp optimizer
-   _Adam_:  Adam optimizer

## +) **Callbacks**

Khi models chúng ta lớn khi training thì gặp lỗi ta muốn lưu lại models để chạy lại thì ta sử dụng callbacks.

-   _ModelsCheckpoint :_ lưu lại model sau mỗi epoch.
-   _EarlyStopping :_ stop training khi models training không hiệu quả.
-   _ReducaLROnPlateau :_ giảm learning mỗi khi metrics không cải thiện.

+) **Applications**

Chứa các pre-training weight của các model Deep learning nổi tiếng như Xception, VGG16, VGG19, Resnet50, Inceptionv3, InceptionResNetV2, MobileNet, DenseNet, NASNet với cấu trúc chung như sau :

- _preprocess_input_ _:_  dùng để preprocessing input custom same với input của pretraining.

 - _decode_predictions_ _:_ dùng để xem label predict.

- _backends_ _:_ có nghĩa là thay vì keras xây dựng từ đầu các công thức từ đơn giản đến phức tạp, thì nó dùng những thư viện đã xây dựng sẵn rồi và dùng thôi. Giúp tiết kiệm dc thời gian và chí phí. Trong keras có hỗ trợ 3 backend là tensorflow,theano và CNTK.

- _initializers :_ khởi tạo giá trị weight của coeff và bias trước khi training lần lượt _kernel_initializer_ và _bias_initializer_. Mặc định là _glorot_uniform_ phân phối uniform với giá trị 1/ căn (input+output).

- _regularizers_ _:_ Dùng để phạt những coeff nào tác động quá mạnh vào mỗi layer thường dùng là L1 và L2.

- _constraints_ _:_ dùng để thiết lập các điều kiện ràng buộc khi training.

- _visualization_ _:_ giúp chúng ta plot lại cấu trúc mạng neral network.

- _Utils_ _:_ chứa các function cần thiết giúp ta xử lý data nhanh hơn.

- _Normalize_ _:_ chuẩn hóa data theo L2.

- _plot_model_ _:_ giúp chúng ta plot model.

- _to_categorical_ _:_ covert class sang binary class matrix.
