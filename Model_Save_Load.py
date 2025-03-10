# 모델 상태 유지 및 예측 실행 방법
import torch
import torchvision.models as models

# 모델 가중치 저장 및 Load
# Pytorch은 학습한 매개변수를 'state_dict' 에 저장 (일시저장?)
# 해당 상태 값은 'torch.save' 메소드를 통해 저장 가능 (영구저장?)
# -> state_dict 라는 속성에 임시저장 -> torch.save를 통해 파일로 내보내 다음 학습 때 빠른 호출 가능
model = models.vgg16(weights='IMAGENET1K_V1')
    #vgg16 = 이미지 분류를 위해 설계된 딥러닝 모델 , 16개의 레이어 포함 
    #weights = ImageNet 데이터셋에서 학습된 가중치 로드(기존 학습된 결과 저장 == 시간 단축)
torch.save(model.state_dict(), 'model_weights.pth') # model.state_dict() 값을 model.weights.pth 라는 이름으로 저장?

model = models.vgg16() #weights 지정하지 않음 -> 학습되지 않은 모델 생성
model.load_state_dict(torch.load('model_weights.pth')) # 학습되지 않은 model에 load 하므로 기존 학습 정보를 불러올 수 있음
                                                        # -> 시간단축 가능
model.eval()



# 모델도 함께 저장 -> 호환성 부분에서 결함이 있어 권장X
torch.save(model, 'model.pth') #단, 모델 전체를 저장하려면 Class가 정의되어야 함 (ex: NeuralNetwork)
# model = torch.load('model.pth')
model = torch.load('model.pth', weights_only=False) #pytorch 2.6부터 torch.load에서 weights_only=True 기본 설정되어 있어, False로 지정해줘야 함