import os
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import time
import numpy as np
import pandas as pd
import pymysql
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#pip install yellowbrick
from datetime import datetime
from sklearn.model_selection import train_test_split
import itertools
from sqlalchemy import create_engine
import sqlalchemy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

#%%
# 2020년 데이터 #

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_1 = "SELECT * FROM cargo_db.tb_cargomylist_2020_1 ;"
cursor.execute(sql_1)
g_1 = cursor.fetchall()
tb_cargomylist_2020_1 = pd.DataFrame(g_1)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_2 = "SELECT * FROM cargo_db.tb_cargomylist_2020_2 ;"
cursor.execute(sql_2)
g_2 = cursor.fetchall()
tb_cargomylist_2020_2 = pd.DataFrame(g_2)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_3 = "SELECT * FROM cargo_db.tb_cargomylist_2020_3_2 ;"
cursor.execute(sql_3)
g_3 = cursor.fetchall()
tb_cargomylist_2020_3_2 = pd.DataFrame(g_3)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', #
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_4 = "SELECT * FROM cargo_db.tb_cargomylist_2020_4_2 ;"
cursor.execute(sql_4)
g_4 = cursor.fetchall()
tb_cargomylist_2020_4_2 = pd.DataFrame(g_4)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_5 = "SELECT * FROM cargo_db.tb_cargomylist_2020_5 ;"
cursor.execute(sql_5)
g_5 = cursor.fetchall()
tb_cargomylist_2020_5 = pd.DataFrame(g_5)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_6 = "SELECT * FROM cargo_db.tb_cargomylist_2020_6 ;"
cursor.execute(sql_6)
g_6 = cursor.fetchall()
tb_cargomylist_2020_6 = pd.DataFrame(g_6)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_7 = "SELECT * FROM cargo_db.tb_cargomylist_2020_7 ;"
cursor.execute(sql_7)
g_7 = cursor.fetchall()
tb_cargomylist_2020_7 = pd.DataFrame(g_7)
cursor.close()

df1 = tb_cargomylist_2020_1
df2 = tb_cargomylist_2020_2
df3 = tb_cargomylist_2020_3_2
df4 = tb_cargomylist_2020_4_2
df5 = tb_cargomylist_2020_5
df6 = tb_cargomylist_2020_6
df7 = tb_cargomylist_2020_7

# 데이터 합치기(행 기준: 위/아래로 합치기)
df_all_2020 = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)

#%%
# 2021년 데이터 #

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_1 = "SELECT * FROM cargo_db.tb_cargomylist_2021_1 ;"
cursor.execute(sql_1)
g_1 = cursor.fetchall()
tb_cargomylist_2021_1 = pd.DataFrame(g_1)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_2 = "SELECT * FROM cargo_db.tb_cargomylist_2021_2 ;"
cursor.execute(sql_2)
g_2 = cursor.fetchall()
tb_cargomylist_2021_2 = pd.DataFrame(g_2)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_3 = "SELECT * FROM cargo_db.tb_cargomylist_2021_3 ;"
cursor.execute(sql_3)
g_3 = cursor.fetchall()
tb_cargomylist_2021_3 = pd.DataFrame(g_3)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', #
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_4 = "SELECT * FROM cargo_db.tb_cargomylist_2021_4 ;"
cursor.execute(sql_4)
g_4 = cursor.fetchall()
tb_cargomylist_2021_4 = pd.DataFrame(g_4)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_5 = "SELECT * FROM cargo_db.tb_cargomylist_2021_5 ;"
cursor.execute(sql_5)
g_5 = cursor.fetchall()
tb_cargomylist_2021_5 = pd.DataFrame(g_5)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_6 = "SELECT * FROM cargo_db.tb_cargomylist_2021_6 ;"
cursor.execute(sql_6)
g_6 = cursor.fetchall()
tb_cargomylist_2021_6 = pd.DataFrame(g_6)
cursor.close()

freight_db = pymysql.connect(
    user='root', 
    passwd='0000', 
    host='127.0.0.1', 
    db='cargo_db',  # 불러올 스키마
    charset='utf8'
)
cursor = freight_db.cursor(pymysql.cursors.DictCursor)
sql_7 = "SELECT * FROM cargo_db.tb_cargomylist_2021_7 ;"
cursor.execute(sql_7)
g_7 = cursor.fetchall()
tb_cargomylist_2021_7 = pd.DataFrame(g_7)
cursor.close()

df1 = tb_cargomylist_2021_1
df2 = tb_cargomylist_2021_2
df3 = tb_cargomylist_2021_3
df4 = tb_cargomylist_2021_4
df5 = tb_cargomylist_2021_5
df6 = tb_cargomylist_2021_6
df7 = tb_cargomylist_2021_7

# 데이터 합치기(행 기준: 위/아래로 합치기)
df_all_2021 = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)

#%%
# 2020년, 2021년 데이터 합치기

df_all_2020.shape # (4265019,80)
df_all_2021.shape # (2502009,81)

df_all = pd.concat([df_all_2020, df_all_2021], axis=0)

df_all.shape # (6767028,81)


#%%
## 톤수별로 데이터 추출 ##
X_y = df_all

# # 25톤
# X_y_2500 = X_y[X_y['intCarSizecode']==2500]
# X_y_2500.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_2500.csv', encoding='cp949')

# # 11톤
# X_y_1100 = X_y[X_y['intCarSizecode']==1100]
# X_y_1100.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1100.csv', encoding='cp949')

# # 18톤
# X_y_1800 = X_y[X_y['intCarSizecode']==1800]
# X_y_1800.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1800.csv', encoding='cp949')

# # 14톤
# X_y_1400 = X_y[X_y['intCarSizecode']==1400]
# X_y_1400.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1400.csv', encoding='cp949')

# # 22톤
# X_y_2200 = X_y[X_y['intCarSizecode']==2200]
# X_y_2200.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_2200.csv', encoding='cp949')

# # 5톤축
# X_y_501 = X_y[X_y['intCarSizecode']==500]
# X_y_501.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_501.csv', encoding='cp949')

# # 16톤
# X_y_1600 = X_y[X_y['intCarSizecode']==1600]
# X_y_1600.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1600.csv', encoding='cp949')

# # 5톤
# X_y_500 = X_y[X_y['intCarSizecode']==500]
# X_y_500.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_500.csv', encoding='cp949')

# # 18톤축
# X_y_1801 = X_y[X_y['intCarSizecode']==1801]
# X_y_1801.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1801.csv', encoding='cp949')

# # 기타
# X_y_9999 = X_y[X_y['intCarSizecode']==9999]
# X_y_9999.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_9999.csv', encoding='cp949')

# # 14톤축
# X_y_1401 = X_y[X_y['intCarSizecode']==1401]
# X_y_1401.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1401.csv', encoding='cp949')

# # 11톤축
# X_y_1101 = X_y[X_y['intCarSizecode']==1101]
# X_y_1101.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1101.csv', encoding='cp949')

# # 9.5톤
# X_y_950 = X_y[X_y['intCarSizecode']==950]
# X_y_950.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_950.csv', encoding='cp949')

# # 502
# X_y_502 = X_y[X_y['intCarSizecode']==502]
# X_y_502.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_502.csv', encoding='cp949')

# # 8톤
# X_y_800 = X_y[X_y['intCarSizecode']==800]
# X_y_800.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_800.csv', encoding='cp949')

# # 1톤
# X_y_100 = X_y[X_y['intCarSizecode']==100]
# X_y_100.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_100.csv', encoding='cp949')

# # 16톤축
# X_y_1601 = X_y[X_y['intCarSizecode']==1601]
# X_y_1601.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_1601.csv', encoding='cp949')

# # 2.5톤
# X_y_250 = X_y[X_y['intCarSizecode']==250]
# X_y_250.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_250.csv', encoding='cp949')

# # 3.5톤
# X_y_350 = X_y[X_y['intCarSizecode']==350]
# X_y_350.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_24.csv', encoding='cp949')

# # 103
# X_y_103 = X_y[X_y['intCarSizecode']==103]
# X_y_103.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_103.csv', encoding='cp949')

# # 140
# X_y_140 = X_y[X_y['intCarSizecode']==140]
# X_y_140.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_140.csv', encoding='cp949')


#%%
## 조건 주기.
X_y.shape      # (6767028, 81)

# 1) 배차완료.
X_y_done = X_y[X_y['intStatus']==3] 
X_y_done.shape # (3136497, 81)

# 2) 총금액 0이하 제거.
X_y_done = X_y_done[ X_y_done['intTotalPrice']>0 ]
X_y_done.shape # (363540, 81)

# 3) 실거리 0이하 제거.
X_y_done = X_y_done[ X_y_done['intDistance']>0 ]
X_y_done.shape # (349816, 81)

# 4) 실거리 600초과 제거.
X_y_done = X_y_done[ X_y_done['intDistance']<=600 ] 
X_y_done.shape # (329532, 81)


#%%
# 조건 준 데이터 수 확인(톤수별).
X_y_done['intCarSizecode'].value_counts()

#%%
# 8톤 데이터 => 분석용으로 이용!
X_y_done_800 = X_y_done[X_y_done['intCarSizecode']==800]
X_y_done_800.to_csv(r'C:\Users\KOREA\Desktop\이은주\화물\data\t_800_done.csv', encoding='cp949')

X_y_done_800.shape # (723,81)
X_done_800 = X_y_done_800

#%% # 데이터 가져오기 #

# read
X_done_800 = pd.read_csv('C:/Users/KOREA/Desktop/이은주/화물/data/t_800_done2.csv',encoding='cp949')
X_done_800.shape # (723,87)
X_done_800.columns

# 변수 선택
X = X_done_800[[#'intStatus'# *1:배차전,2:배차대기,3:배차완료,4:상차완료,5:하차완료,6:배차취소,7:배차실패
      #,'bShare'# 1:공유,0:공유안됨(자차)
      'strUpZone' # *상차지역코드
      ,'strDwZone' # *하차지역코드
      ,'intCarTypecode' # *차량종류코드
      #,'intCarSizecode' # *차량톤수코드
      ,'intCarUpType' # 적재구분 1:독차,2:혼적
      ,'intPaytypecode' # 결제방법코드
      ,'intTotalPrice' # 총금액(관제에서사용)
      ,'intMemberPrice' # *기사운임
      ,'intArrangePrice' # 주선수수료-선착불
      ,'intDistance' # *실거리
      ,'intDefaultPrice' # 표준운임
      #,'intCancelLock' # 1:lock,0:nolock
      #,'intRegComCode' #
      #,'intActApp' # 1:관제,2:모바일,3:모바일V1,4:빽통,5:홈페이지,6:늘푸른,7:CJ대한통운,8:SKOLO
      ,'intUpDateType' # *1:당상,2:낼상,3:월상,4:예약
      ,'intDwDateType' # *1:당상,2:낼상,3:월상,4:예약
      #,'intDelivery' # 택배화물 0:아님,1:택배포함
      #,'intMemberCarTypeCode' # 차주카타입 코드
      #,'intMemberCarSizeCode' # 차주카톤 코드
      ,'intFeeSend' # -1:NULL및공백,1:N(운송료못받음),2:Y(운송료받음),3:B(운송료못받음환불),0:C(받음확정) 
      ,'weekUp'
      ,'weekDown'
      ,'timeUp'
      ,'timeDown'
      ]]

# colname = ['strUpZone', 'strDwZone', 'intCarTypecode', 'intCarUpType', 'intPaytypecode', 
#            'intTotalPrice', 'intMemberPrice', 'intArrangePrice', 'intDistance', 
#            'intDefaultPrice', 'intUpDateType', 'intDwDateType', 'intFeeSend',
#            'weekUp', 'weekDown', 'timeUp', 'timeDown', 'FeeKm']

X.shape # (723, 17)

# 'price_per_km': km당 요금 컬럼 생성.
X['FeeKm'] = X['intTotalPrice']/X['intDistance']

X.shape # (723,18)

#%%

# 결측치가 하나라도 있다면 열 제거.
# X = X.dropna(axis=1)

# path
os.chdir(r'C:\Users\KOREA\Desktop\이은주\화물\DeepClustering-master_0511')
#%%
cluster_num = 5

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			#nn.Linear(28 * 28, 500),
			nn.Linear(X.shape[1], 10), #변수 14개.
            nn.ReLU(True),
			#nn.Linear(500, 500),
            nn.Linear(10, 10),
			nn.ReLU(True),
			#nn.Linear(500, 500),
			nn.Linear(10, 10),
            nn.ReLU(True), 
			#nn.Linear(500, 2000),
			nn.Linear(10, 40),
            nn.ReLU(True),
			#nn.Linear(2000, 10))
            nn.Linear(40, cluster_num))
		self.decoder = nn.Sequential(
			#nn.Linear(10, 2000),
            nn.Linear(cluster_num, 40),
			nn.ReLU(True),
			#nn.Linear(2000, 500),
			nn.Linear(40, 10),
            nn.ReLU(True),
			#nn.Linear(500, 500),
            nn.Linear(10, 10),
			nn.ReLU(True),
			#nn.Linear(500, 500),
			nn.Linear(10, 10),
            nn.ReLU(True),
			#nn.Linear(500, 28 * 28))
            nn.Linear(10, X.shape[1]))
		self.model = nn.Sequential(self.encoder, self.decoder)
	def encode(self, x):
		return self.encoder(x)

	def forward(self, x):
	    x = self.model(x)
	    return x


class ClusteringLayer(nn.Module):
	#def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
    def __init__(self, n_clusters=cluster_num, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
            self.n_clusters,
            self.hidden,
            dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)
    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist

class DEC(nn.Module):
	#def __init__(self, n_clusters=10, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
    def __init__(self, n_clusters=cluster_num, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x) 
        return self.clusteringlayer(x)

    def visualize(self, epoch,x):  # plot 그리기
        fig = plt.figure()
        # 단일 정수로 인코딩 된 서브 플롯 그리드 매개 변수. 111: 1×1 그리드, 첫 번째 서브 플롯.
        ax = plt.subplot(111) 
        # x: Encoder가 생성한 데이터.
        x = self.autoencoder.encode(x).detach()
        # x가 cuda 디바이스로 올라갔을 때 torch에서 바로 numpy로 바꿀 수 없기 때문에 cpu로 보내준후 바꾸어 주어야한다.
        x = x.cpu().numpy()#[:2000] [:40]
        # 생성한 x를 2차원 공간상에 시각화하는 t-SNE 학습
        x_embedded = TSNE(n_components=2).fit_transform(x) 
        
        plt.scatter(x_embedded[:,0], x_embedded[:,1]) # x축,y축 변수 수정(상차지와 거리?)
        fig.savefig('plots/8t_{}.png'.format(epoch))
        plt.close(fig)

def add_noise(img):
	noise = torch.randn(img.size()) * 0.2
	noisy_img = img + noise
	return noisy_img

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")

def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(dataset=data,
                              batch_size=128, 
                              shuffle=True)
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img  = data.float()
            noisy_img = add_noise(img)
            noisy_img = noisy_img.to(device)
            img = img.to(device)
		    # ===================forward=====================
            output = model(noisy_img)
            output = output.squeeze(1)
		    #output = output.view(output.size(0), 28*28)
            output = output.view(output.size(0), X.shape[1])
            loss = nn.MSELoss()(output, img)
		    # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
		# ===================log========================
        print('epoch [{}/{}], MSE_loss:{:.4f}'
	      .format(epoch + 1, num_epochs, loss.item()))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)

def acc(y_true, y_pred): #추가
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true, y_pred)
    
def train(**kwargs):
    data = x #수정
    labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    features = []
    train_loader = DataLoader(dataset=data,
                              batch_size=128, 
                              shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach().cpu())
    features = torch.cat(features)
	# ============K-means=======================================
	#kmeans = KMeans(n_clusters=10, random_state=0).fit(features)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
	# =========================================================
    y_pred = kmeans.predict(features)
    accuracy = acc(y.cpu().numpy(), y_pred)
    print('Initial Accuracy: {}'.format(accuracy))

    loss_function = nn.KLDivLoss(size_average=False)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)
    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        img = img.to(device)
        output = model(img)
        target = model.target_distribution(output).detach()
        out = output.argmax(1)
        if epoch % 20 == 0:
            print('plotting')
            dec.visualize(epoch, img)
        loss = loss_function(output.log(), target) / output.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
        row.append([epoch, accuracy])
        print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, num_epochs, accuracy, loss))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)

    df = pd.DataFrame(row, columns=['epochs', 'accuracy'])
    y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])
    df.to_csv('log.csv')
    y_pred_df.to_csv('log_y_pred.csv')
#%%
# 데이터 불러오는 부분. -> 수정 필요! 
def load_data():
    # the data, shuffled and split between train and test sets
# 	train = MNIST(root='./data/',
# 	            train=True, 
# 	            transform=transforms.ToTensor(),
# 	            download=True)

# 	test = MNIST(root='./data/',
# 	            train=False, 
# 	            transform=transforms.ToTensor())
    

	#x_train, y_train = train.train_data, train.train_labels
    x_train = X.iloc[0:len(X.sample(frac=0.8))]
    x_train = pd.get_dummies(x_train) 
    x_train = torch.from_numpy(x_train.values) # numpy를 Torch tensor로 변환.
    #x_train = transforms.ToTensor()
    
	#x_test, y_test = test.test_data, test.test_labels
    x_test = X.iloc[len(X.sample(frac=0.8)):]
    x_test = pd.get_dummies(x_test) 
    x_test = torch.from_numpy(x_test.values)
    #x_test = transforms.ToTensor()
    
    #수정(y만들기)
    Y = [[0] for y in range(len(X))]
    y_train = Y[0:len(X.sample(frac=0.8))]
    y_test = Y[len(X.sample(frac=0.8)):]
    y_train = torch.from_numpy(np.array(y_train))
    y_test = torch.from_numpy(np.array(y_test))
    ##
    x = torch.cat((x_train, x_test), 0)
    y = torch.cat((y_train, y_test), 0)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)  # x를 255로 나누기.
    print('8t samples', x.shape)
    return x, y #수정

#%%
if __name__ == '__main__':
    import argparse
    
    cluster_num = 5

    parser = argparse.ArgumentParser(description='train',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=128, type=int)  # 하이퍼파라미터 줄이면 빨리 결과 볼 수 있음.
    parser.add_argument('--pretrain_epochs', default=20, type=int)
    parser.add_argument('--train_epochs', default=50000, type=int)
    parser.add_argument('--save_dir', default='saves')
    args = parser.parse_args()
    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    # x = load_mnist() 
    x, y = load_data()
    autoencoder = AutoEncoder().to(device)
    import random
    kk=str(round(random.random(),cluster_num)*10000)

    ae_save_path = 'saves/freightsim_autoencoder'+kk+'.pth' #sim_autoencoder.pth'수정

    if os.path.isfile(ae_save_path):
        print('Loading {}'.format(ae_save_path))
        checkpoint = torch.load(ae_save_path)
        autoencoder.load_state_dict(checkpoint['state_dict']) ##여기가 에러남-해결.
    else:
        print("=> no checkpoint found at '{}'".format(ae_save_path))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    pretrain(data=x, model=autoencoder, num_epochs=epochs_pre, savepath=ae_save_path, checkpoint=checkpoint)
	

    dec_save_path='saves/freightdec'+kk+'.pth' #수정dec.pth
	#dec = DEC(n_clusters=10, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
    dec = DEC(n_clusters=cluster_num, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
    if os.path.isfile(dec_save_path):
        print('Loading {}'.format(dec_save_path))
        checkpoint = torch.load(dec_save_path)
        dec.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(dec_save_path))
        checkpoint = {
			"epoch": 0,
		    "best": float("inf")
        }
    train(data=x, labels=y, model=dec, num_epochs=args.train_epochs, savepath=dec_save_path, checkpoint=checkpoint)
	
    
## check the model summary ##    
# from torchvision import models
# model = models.vgg16()
# print(model)
	
    	