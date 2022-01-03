from src.data_ai import generate_models
import argparse

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    default='dataset_teste_selecao.csv',
                    help='path to csv data')

    args = vars(ap.parse_args())
    generate_models(args['input'])
    print('[INFO] Modelos gerado e dados de testes salvos')

if __name__ == '__main__':
    main()