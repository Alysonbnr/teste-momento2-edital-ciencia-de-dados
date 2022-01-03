from src.data_process import data_visualize
import argparse

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    default='dataset_teste_selecao.csv',
                    help='path to csv data')

    args = vars(ap.parse_args())
    data_visualize(args['input'])

if __name__ == '__main__':
    main()