from src.data_ai import data_prediction, models_evaluate
import joblib
import argparse
import numpy as np

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--test_data',
                    default='src/data/test_data.npy',
                    help='path to test data')
    ap.add_argument('-l', '--label_test_data',
                    default='src/data/label_test_data.npy',
                    help='path to label test data')
    ap.add_argument('-rdf', '--random_forest',
                    default='src/data/random_forest.joblib',
                    help='path to random forest model')
    ap.add_argument('-mlp', '--mlp',
                    default='src/data/mlp.joblib',
                    help='path to mlp model')
    ap.add_argument('-knn', '--knn',
                    default='src/data/knn.joblib',
                    help='path to knn model')
    ap.add_argument('-tree', '--tree',
                    default='src/data/tree.joblib',
                    help='path to decision tree model')
    ap.add_argument('-gnb', '--gausiannb',
                    default='src/data/GaussianNB.joblib',
                    help='path to GaussianNB model')
    ap.add_argument('-km', '--kmeans',
                    default='src/data/kmeans.joblib',
                    help='path to kmeans model')
    ap.add_argument('-svm', '--svm',
                    default='src/data/svm.joblib',
                    help='path to svm model')
    ap.add_argument('-sgdc', '--sgdc',
                    default='src/data/sgdc.joblib',
                    help='path to sgdc model')


    args = vars(ap.parse_args())
    models_list = [joblib.load(args['knn']),joblib.load(args['mlp']),joblib.load(args['tree']),\
                   joblib.load(args['random_forest']),joblib.load(args['svm']) , joblib.load(args['sgdc']), \
                   joblib.load(args['gausiannb']),joblib.load(args['kmeans'])]
    test_data = np.load(args['test_data'])
    label_test_data = np.load(args['label_test_data'])

    resp = data_prediction(models_list,test_data)
    models_evaluate(resp,label_test_data)


if __name__ == '__main__':
    main()

