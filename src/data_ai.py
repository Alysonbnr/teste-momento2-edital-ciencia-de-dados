import random
import joblib
import numpy as np
from src.data_process import data_preprocess, data_adjust
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics,svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree




def train_test_split(data, train_percent=.85, test_percent=.15):

    num_of_cases, num_of_atributes = data.shape
    size_train = int(train_percent * num_of_cases)
    size_test = int(test_percent * num_of_cases)
    train_data = data[0:size_train]
    label_train_data = train_data[:,-1]
    train_data = train_data[:,0:-1]
    test_data = data[size_train:size_train+size_test]
    label_test_data = test_data[:,-1]
    test_data = test_data[:,0:-1]

    return train_data,label_train_data, test_data,label_test_data

def gen_data(csv_path):

    _, _, all_data = data_preprocess(csv_path)
    all_data_array = data_adjust(all_data)
    random.shuffle(all_data_array)
    train_data,label_train_data, test_data,label_test_data= train_test_split(all_data_array)
    return  train_data,label_train_data , test_data,label_test_data

def generate_random_forest_model(train_data,label_train_data):

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(train_data,label_train_data)
    joblib.dump(clf, "src/data/random_forest.joblib")


def generate_MLP_model(train_data,label_train_data):
    classifier = MLPClassifier(hidden_layer_sizes=(90,100,100), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(train_data, label_train_data)
    joblib.dump(classifier, "src/data/mlp.joblib")

def generate_knn_model(train_data,label_train_data):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, label_train_data)
    joblib.dump(knn, "src/data/knn.joblib")

def generate_svm_model(train_data,label_train_data):
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_data, label_train_data)
    joblib.dump(clf, "src/data/svm.joblib")

def generate_kmeans_model(train_data):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train_data)
    joblib.dump(kmeans, "src/data/kmeans.joblib")

def generate_SGDC_model(train_data,label_train_data):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    clf.fit(train_data, label_train_data)
    joblib.dump(clf, "src/data/sgdc.joblib")

def generate_naive_bayes_model(train_data,label_train_data):
    gnb = GaussianNB()
    gnb.fit(train_data, label_train_data)
    joblib.dump( gnb, "src/data/GaussianNB.joblib")

def generate_decision_tree_model(train_data,label_train_data):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, label_train_data)
    joblib.dump( clf, "src/data/tree.joblib")

def save_teste_data(test_data,label_test_data):
    np.save('src/data/test_data.npy',test_data)
    np.save('src/data/label_test_data.npy',label_test_data)

def data_prediction(models_list,teste_data):

    resp_knn = models_list[0].predict(teste_data)
    mlp_resp = models_list[1].predict(teste_data)
    tree_resp = models_list[2].predict(teste_data)
    rdf_resp = models_list[3].predict(teste_data)
    svm_resp = models_list[4].predict(teste_data)
    sgdc_resp = models_list[5].predict(teste_data)
    gnb_resp = models_list[6].predict(teste_data)
    kmean_resp = models_list[7].predict(teste_data)

    return resp_knn,mlp_resp,tree_resp,rdf_resp,svm_resp,sgdc_resp,gnb_resp,kmean_resp

def models_evaluate(resp_list,label_test):
   print('\n [INFO]- A tabela abaixo indica a acurácia do auxilio ao diagnóstico de COVID-19 por 8 classificadores com base em sintomas e outros dados extraídos de pacientes \n')
   print(" KNN Accuracy:", metrics.accuracy_score(resp_list[0], label_test))
   print(" MLP Accuracy:", metrics.accuracy_score(resp_list[1], label_test))
   print(" DECISION TREE Accuracy:", metrics.accuracy_score(resp_list[2], label_test))
   print(" RANDOM FOREST Accuracy:", metrics.accuracy_score(resp_list[3], label_test))
   print(" SVM Accuracy:", metrics.accuracy_score(resp_list[4], label_test))
   print(" SGDC Accuracy:", metrics.accuracy_score(resp_list[5], label_test))
   print(" GAUSSIAN NAIVE BAYES Accuracy:", metrics.accuracy_score(resp_list[6], label_test))
   print(" KMEANS Accuracy:", metrics.accuracy_score(resp_list[7], label_test))
   print('\n Atibutos analisados foram: \n\n diabetes,obesidade,doenca_pulmonar,sexo,tosse_seca_ou_produtiva,cefaleia, paciente_chegou_com_suporte_respiratorio,idade')

def generate_models(csv_path):

    train_data,label_train_data,test_data,label_test_data = gen_data(csv_path)
    generate_random_forest_model(train_data,label_train_data)
    generate_MLP_model(train_data,label_train_data)
    generate_knn_model(train_data,label_train_data)
    generate_svm_model(train_data,label_train_data)
    generate_kmeans_model(train_data)
    generate_SGDC_model(train_data,label_train_data)
    generate_naive_bayes_model(train_data,label_train_data)
    generate_decision_tree_model(train_data,label_train_data)
    save_teste_data(test_data,label_test_data)


