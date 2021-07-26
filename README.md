# My-CNN-based-Image-Classification-in-PyTorch
Practice following the tutorial

본 프로젝트에서는 **YongHye Kwon(@developer0hye)** 님의 튜토리얼을 따라 **Image Classifier**를 만들어본다.

이를 통해 다음 방법들을 익힌다.

* **CNN(Convolutional Neural Network)** 기반의 **Image Classifier** 모델 설계 방법
* 기존의 데이터셋(MNIST, CIFAR-10 등)이 아닌 **Custom Dataset**(개인이 수집한 데이터셋)을 처리하기 위한 **PyTorch**의 **Dataset** 및 **DataLoader** 클래스 사용법

## Setup

### Custom Dataset
본 프로젝트에서는 게임 **"앙상블 스타즈"** 의 등장 인물인 **"이즈미"** 와 **"코가"** 를 분류해본다.

#### 앙상블 스타즈
![EnsenbleStars](https://user-images.githubusercontent.com/87509229/126755746-6ae341e9-b880-4e76-9e98-c72b2dd0957e.jpg)

#### 이즈미
![izumi](https://user-images.githubusercontent.com/87509229/126755827-2f6cb576-a50a-4380-8f67-77c759f8a669.png)

#### 코가
![koga](https://user-images.githubusercontent.com/87509229/126755835-312d4b6b-38b6-4f7e-b3ef-91df9782b19a.png)

### Examples
이즈미와 코가 각각 30장씩 얼굴 이미지 크롭
![datasetcapture_izumi](https://user-images.githubusercontent.com/87509229/126755155-1c983f3d-3289-419a-9b54-466fef108c1c.PNG)
![datasetcapture_koga](https://user-images.githubusercontent.com/87509229/126755157-96d5dec9-00e0-4969-86a0-a0c493e46f18.PNG)

### Struct of Directory
본 프로젝트에서는 **Custom Dataset**에 대한 처리를 보다 쉽게 하기 위해 다음과 같이 main.py 파일과 **Custom Dataset**이 동일한 경로에 있다고 가정한다.

```pyton

datasets/
  train/
    izumi/
      *.png
    koga/
      *.png
  test/
    izumi/
      *.png
    koga/
      *.png

main.py
```

## Data Loading and Processing
PyTorch가 제공하는 Dataset, DataLoader 클래스를 사용하면 데이터 처리를 용이하게 할 수 있다.

**Dataset**클래스는 torch.utils.data.Dataset에 정의된 추상 클래스(Abstract Classs)이다.
사용자는 Dataset 클래스를 상속받는 클래스를 정의함으로써 **Custom Dataset**을 다룰 수 있다.

**DataLoader**클래스는 **Dataset** 클래스를 상속받는 **Custom Dataset**에 정의된 작업에 따라 데이터를 읽어온다.
이때, 인자로 전달되는 플래그를 통해 데이터를 읽어올 **배치(Batch) 크기**, **병렬 처리 여부**, **데이터 셔플(Shuffle) 여부** 등의 작업을 설정할 수 있다.

이하 **Dataset** 클래스를 상속받는 커스텀 클래스를 **Dataset** 클래스라고 통칭한다.

### Dataset Class & DataLoader Class
**Custom Dataset**을 읽기 위해서는 기본적으로 다음 3가지 함수를 작성해야 한다.

* __init__()
* __getitem__()
* __len__()


`__init__()`: 클래스 생성자로, 데이터에 대한 Transform(데이터 형 변환, Augmentation 등)을 설정하고 데이터를 읽기 위한 기초적인 초기화 작업들을 수행할 수 있게끔 정의하면 된다.

`__getitem__()`: 특정 데이터를 읽고자 할 때 사용할 함수이다. 우리가 준비한 Custom Dataset에서 해당 데이터를 읽고 반환한다. 즉, 사용자는 읽어온 데이터로 어떤 작업을 수행하려는건지 파악하고 그에 맞춰 함수의 반환값을 사용자화하면 된다.  

`__len__()`: 클래스를 구성하는 원소의 개수를 얻을 수 있는 함수이다. 본 프로젝트의 Dataset 클래스는 이미지의 집합이므로, 이 `__len__()`함수는 데이터셋에 존재하는 이미지 데이터의 개수를 반환한다.

위 함수들은 모두 파이썬이 제공하는 빌트인 함수로서, 매직 메서드(magic method) 또는 던더 메서드(dunder method; double-under)라고 부른다. 이외에도 `__add__()`,  `__sub__()` 등 다양한 메서드가 존재한다. 
https://docs.python.org/ko/3.7/reference/datamodel.html#special-method-names
에서 자세히 볼 수 있다.

### Programming

#### Declaration & Definition class
```python
class MyImageDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.ReadDataset()
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length

    def ReadDataset(self):
        img_files = []
        labels = []

        class_names = os.walk(self.dataset_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.dataset_path, class_name)
            img_files_path = os.walk(img_dir).__next__()[2]

            for img_file_path in img_files_path:
                img_file = os.path.join(img_dir, img_file_path)
                img = Image.open(img_file)
                if img is not None:
                    img_files.append(img_file)
                    labels.append(label)

        return img_files, labels, len(labels), len(class_names)
```

## MileStone

### How to implement NN with PyTorch

