
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def csv_to_list(data_csv):
    all_data_list = []
    death_include_list = []
    clf_list = []

    for coriza,tosse_seca_ou_produtiva, calafrios,	febre,	dispneia,\
        fadiga,	anorexia,	mialgia,	astenia,	dor_de_garganta,\
        congestao_nasal,	cefaleia,	diarreia,	nausea,	vomitos,\
        anosmia,	ageusia,	diabetes,\
        obesidade,	hipertensao_arterial,	doenca_cardiaca,	doenca_pulmonar,\
        doenca_reumatologica,	doenca_autoimune,	doenca_renal_cronica,\
        doenca_hepatica_cronica,	doenca_neurologica,	paciente_chegou_com_suporte_respiratorio,\
        sexo,hemorragia,bairro,idade,complicacao_neurologica,desfecho,tipo_caso\
        in zip(data_csv['coriza'],data_csv['tosse_seca_ou_produtiva'],data_csv['calafrios'], \
            data_csv['febre'],data_csv['dispneia'],data_csv['fadiga'], \
               data_csv['anorexia'],data_csv['mialgia'],data_csv['astenia'], \
               data_csv['dor_de_garganta'],data_csv['congestao_nasal'],data_csv['cefaleia'], \
               data_csv['diarreia'],data_csv['nausea'],data_csv['vomitos'], data_csv['anosmia'], \
               data_csv['ageusia'], \
               data_csv['diabetes'],data_csv['obesidade'],data_csv['hipertensao_arterial'],data_csv['doenca_cardiaca'], \
               data_csv['doenca_pulmonar'],data_csv['doenca_reumatologica'],\
               data_csv['doenca_autoimune'],data_csv['doenca_renal_cronica'], \
               data_csv['doenca_hepatica_cronica'],data_csv['doenca_neurologica'],\
               data_csv['paciente_chegou_com_suporte_respiratorio'],data_csv['sexo'], data_csv['hemorragia'], data_csv['bairro'],\
               data_csv['idade'],data_csv['complicacao_neurologica'],data_csv['desfecho'],data_csv['tipo_caso_à_admissão']):

        all_data_list.append([coriza,tosse_seca_ou_produtiva, calafrios,	febre,	dispneia, \
                          fadiga,	anorexia,	mialgia,	astenia,	dor_de_garganta, \
                          congestao_nasal,	cefaleia,	diarreia,	nausea,	vomitos, \
                          anosmia,	ageusia,	diabetes, \
                          obesidade,	hipertensao_arterial,	doenca_cardiaca,	doenca_pulmonar, \
                          doenca_reumatologica,	doenca_autoimune,	doenca_renal_cronica, \
                          doenca_hepatica_cronica,	doenca_neurologica,	paciente_chegou_com_suporte_respiratorio,sexo,bairro,idade,tipo_caso])

        death_include_list.append([coriza,tosse_seca_ou_produtiva, calafrios,	febre,	dispneia, \
                                    fadiga,	anorexia,	mialgia,	astenia,	dor_de_garganta, \
                                    congestao_nasal,	cefaleia,	diarreia,	nausea,	vomitos, \
                                    anosmia,	ageusia,	diabetes, \
                                    obesidade,	hipertensao_arterial,	doenca_cardiaca,	doenca_pulmonar, \
                                    doenca_reumatologica,	doenca_autoimune,	doenca_renal_cronica, \
                                    doenca_hepatica_cronica,	doenca_neurologica,	paciente_chegou_com_suporte_respiratorio,\
                                    sexo,idade,desfecho,tipo_caso])

        clf_list.append([diabetes,obesidade,doenca_pulmonar,sexo,tosse_seca_ou_produtiva,cefaleia,paciente_chegou_com_suporte_respiratorio,idade,tipo_caso])


    return all_data_list , death_include_list , clf_list

def normalize_age(data_array):

    for age in range(len(data_array)):
        data_array[age] = float(float(data_array[age])/100)

    return data_array

def data_adjust(data_list):
    data_array = np.array(data_list)
    data_array_return = np.zeros_like(data_array,dtype=float)
    id_sim = np.where(data_array=='Sim')
    id_masculino = np.where(data_array=='Masculino')
    id_feminino = np.where(data_array=='Feminino')
    id_nao =  np.where(data_array=='Não')
    id_caso_suspeito =  np.where(data_array=='Caso suspeito')
    id_caso_confirmado =  np.where(data_array=='Caso confirmado')

    data_array_return [id_sim] = int(1)
    data_array_return [id_nao] = int(0)
    data_array_return [id_masculino] = int(1)
    data_array_return [id_feminino] = int(0)
    data_array_return [id_caso_suspeito] = int(0)
    data_array_return [id_caso_confirmado]=  int(1)
    data_array_return [:,-2] = normalize_age(data_array[:,-2])
    data_array_return  = data_array_return [0:-2,:]
    return data_array_return

def plot_stack_bar_covid(covid,not_covid):


    covid = np.array(covid)
    not_covid = np.array(not_covid)
    bars = ['Casos Totais','Homens',',Mulheres','Diabéticos(as)','Idade Superior \n a 60 anos','Obesidade','Chegou com \n suporte respiratório']
    plt.figure(figsize=(10,10))
    plt.bar( bars, covid, color = 'black',bottom = not_covid)
    plt.bar( bars, not_covid, color = 'grey')

    for i,v in enumerate(not_covid):
        plt.text(i, v - 50, str(v), color='white', fontweight='bold')

    for i2,v2 in enumerate(covid):
        plt.text(i2, v2 * 2.5, str(v2), color='red', fontweight='bold')

    plt.xlabel(r"$\bf{Situações}$ "+ r"$\bf{Analisadas}$ ")
    plt.ylabel( r"$\bf{Casos}$ " + r"$\bf{para}$ "+ r"$\bf{Cada}$ "+  r"$\bf{Grupo}$ " )
    plt.title( r"$\bf{Análise}$ "+ r"$\bf{da}$ " + r"$\bf{Aquisição}$ "+ r"$\bf{de}$ " + r"$\bf{Covid-19}$ "+ r"$\bf{Considerando}$ " + r"$\bf{algumas}$ "+ r"$\bf{Características}$ " +  r"$\bf{Específicas}$ "+ r"$\bf{de}$ " + r"$\bf{Cada}$ "+ r"$\bf{Paciente}$")
    plt.legend(('Casos com covid-19 Confirmada', 'Casos de não confirmação de covid-19'))
    r"$\bf{Caractirísticas}$" r"$\bf{Analisadas}$"
    plt.savefig('src/results/Analise de aquisição de covid.png', format='png')


def plot_stack_bar_death(death,not_death):


    death = np.array(death)
    not_death = np.array(not_death)
    bars = ['Casos Totais','Homens',',Mulheres','Diabéticos(as)','Idade Superior \n a 60 anos','Obesidade','Chegou com \n suporte respiratório']
    plt.figure(figsize=(10,10))
    plt.bar( bars, death, color = 'black',bottom = not_death)
    plt.bar( bars, not_death, color = 'grey')

    for i,v in enumerate(not_death):
        plt.text(i, v - 10, str(v), color='white', fontweight='bold')

    for i2,v2 in enumerate(death):
        if i2 == 0:
         plt.text(0, v2 + 80, str(v2), color='red', fontweight='bold')
        if i2 == 2 or i2 == 1:
         plt.text(i2, v2 + 40, str(v2), color='red', fontweight='bold')
        if i2 == 3 or i2 == 4 :
         plt.text(i2, v2 + 28, str(v2), color='red', fontweight='bold')
        if i2 == 5 :
         plt.text(i2, v2 +15, str(v2), color='red', fontweight='bold')
        if i2 == 6 :
         plt.text(i2, v2*2, str(v2), color='red', fontweight='bold')


    plt.xlabel(r"$\bf{Situações}$ "+ r"$\bf{Analisadas}$ ")
    plt.ylabel( r"$\bf{Casos}$ " + r"$\bf{para}$ "+ r"$\bf{Cada}$ "+  r"$\bf{Grupo}$ " )
    plt.title( r"$\bf{Análise}$ "+ r"$\bf{da}$ " + r"$\bf{morte}$ "+ r"$\bf{por}$ " + r"$\bf{Covid-19}$ "+ r"$\bf{Considerando}$ " + r"$\bf{algumas}$ "+ r"$\bf{Características}$ " +  r"$\bf{Específicas}$ "+ r"$\bf{de}$ " + r"$\bf{Cada}$ "+ r"$\bf{Paciente}$")
    plt.legend(('Casos de óbito por covid-19', 'Casos de Alta'))
    r"$\bf{Caractirísticas}$" r"$\bf{Analisadas}$"
    plt.savefig('src/results/Analise de morte por covid.png', format='png')
    plt.show()



def age_analize(age_array,array_adjusted):
    idoso_list_covid = []
    idoso_list_not_covid = []
    for index in range(len(array_adjusted)):
        if float(age_array[index]) > 60 and (array_adjusted[index,-1]) == 'Caso confirmado' :
           idoso_list_covid.append(array_adjusted[index])
        if float(age_array[index]) > 60 and (array_adjusted[index,-1]) == 'Caso suspeito' :
           idoso_list_not_covid.append(array_adjusted[index])

    return len(idoso_list_covid) , len(idoso_list_not_covid)

def death_age_analize(age_array,array_adjusted):
    idoso_list_death = []
    idoso_list_not_death = []
    for index in range(len(array_adjusted)):
        if float(age_array[index]) > 60 and (array_adjusted[index,-1]) == 'Caso confirmado'  and (array_adjusted[index,-2] == 'obito'):
            idoso_list_death.append(array_adjusted[index])
        if float(age_array[index]) > 60 and (array_adjusted[index,-1]) == 'Caso confirmado'and (array_adjusted[index,-2] == 'alta') :
            idoso_list_not_death.append(array_adjusted[index])

    return len( idoso_list_death) , len(idoso_list_not_death)

def data_analize_covid(all_data_list):

  data_array = np.array(all_data_list)
  id_caso_confirmado =  np.where(data_array=='Caso confirmado')
  id_caso_suspeito =  np.where(data_array=='Caso suspeito')

  id_masc =  np.where(data_array =='Masculino')
  id_femi =  np.where(data_array =='Feminino' )
  id_masc_covid = np.where(data_array[id_masc[0]] == 'Caso confirmado')
  id_femi_covid = np.where(data_array[id_femi[0]] == 'Caso confirmado')
  id_masc_not_covid = np.where(data_array[id_masc[0]] == 'Caso suspeito')
  id_femi_not_covid = np.where(data_array[id_femi[0]] == 'Caso suspeito')

  id_diabetes =  np.where(data_array[:,17] =='Sim')
  id_diabetes_covid = np.where(data_array[id_diabetes [0]] == 'Caso confirmado')
  id_diabetes_not_covid = np.where(data_array[id_diabetes [0]] == 'Caso suspeito')

  id_obeso =  np.where(data_array[:,18] =='Sim')
  id_obeso_covid = np.where(data_array[id_obeso  [0]] == 'Caso confirmado')
  id_obeso_not_covid = np.where(data_array[id_obeso  [0]] == 'Caso suspeito')

  id_suporte=  np.where(data_array[:,27] =='Sim')
  id_suporte_covid = np.where(data_array[id_suporte  [0]] == 'Caso confirmado')
  id_suporte_not_covid = np.where(data_array[id_suporte  [0]] == 'Caso suspeito')

  num_suporte_covid = len(id_suporte_covid[0])
  num_suporte_not_covid = len( id_suporte_not_covid[0])
  num_obeso_covid = len(id_obeso_covid[0])
  num_obeso_not_covid = len(id_obeso_not_covid[0])
  num_idoso_covid , num_idoso_not_covid = age_analize(data_array [:,-2],data_array)
  num_diabetes_covid = len(id_diabetes_covid [0])
  num_diabetes_not_covid = len(id_diabetes_not_covid [0])
  num_covid_masc = len(id_masc_covid[0])
  num_covid_masc_not = len(id_masc_not_covid[0])
  num_covid_fem = len(id_femi_covid[0])
  num_covid_fem_not = len(id_femi_not_covid[0])
  num_covid = len(id_caso_confirmado[0])
  num_not_covid = len(id_caso_suspeito[0])


  plot_stack_bar_covid([num_covid,num_covid_masc,num_covid_fem,num_diabetes_covid,num_idoso_covid,\
                        num_obeso_covid,num_suporte_covid],[num_not_covid ,num_covid_masc_not,num_covid_fem_not,\
                                          num_diabetes_not_covid,num_idoso_not_covid,num_obeso_not_covid,num_suporte_not_covid])

def death_data_analize(all_data_list):

    data_array = np.array(all_data_list)
    id_caso_confirmado_death = np.where(np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado'))
    id_caso_confirmado_not_death = np.where( np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado'))

    id_masc_death = np.where(np.logical_and(data_array[:,-4] == 'Masculino',np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado')))
    id_masc_not_death =  np.where(np.logical_and(data_array[:,-4] == 'Masculino',np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado')))

    id_fem_death = np.where(np.logical_and(data_array[:,-4] == 'Feminino',np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado')))
    id_fem_not_death = np.where(np.logical_and(data_array[:,-4] == 'Feminino',np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado')))

    id_diabetes_death = np.where(np.logical_and(data_array[:,17] == 'Sim',np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado')))
    id_diabetes_not_death = np.where(np.logical_and(data_array[:,17] == 'Sim',np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado')))

    id_obeso_death = np.where(np.logical_and(data_array[:,18] == 'Sim',np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado')))
    id_obeso_not_death = np.where(np.logical_and(data_array[:,18] == 'Sim',np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado')))

    id_sp_death = np.where(np.logical_and(data_array[:,27] == 'Sim',np.logical_and(data_array[:,-2] =='obito',data_array[:,-1] == 'Caso confirmado')))
    id_sp_not_death = np.where(np.logical_and(data_array[:,27] == 'Sim',np.logical_and(data_array[:,-2] =='alta',data_array[:,-1] == 'Caso confirmado')))

    num_obeso_covid_death = len(id_obeso_death [0])
    num_obeso_covid_not_death= len(id_obeso_not_death[0])

    num_idoso_death , num_idoso_not_death= death_age_analize(data_array [:,-3],data_array)

    num_diabetes_death = len(id_diabetes_death[0])
    num_diabetes_not_death = len(id_diabetes_not_death[0])

    num_masc_death = len(id_masc_death[0])
    num_masc_not_death = len(id_masc_not_death[0])

    num_fem_death = len(id_fem_death[0])
    num_fem_not_death = len(id_fem_not_death[0])

    num_death = len(id_caso_confirmado_death[0])
    num_not_death = len(id_caso_confirmado_not_death[0])

    num_sp_death = len(id_sp_death[0])
    num_sp_not_death = len(id_sp_not_death[0])


    plot_stack_bar_death([num_death,num_masc_death,num_fem_death,num_diabetes_death,num_idoso_death, \
                          num_obeso_covid_death, num_sp_death],[num_not_death ,num_masc_not_death,num_fem_not_death, \
                                            num_diabetes_not_death,num_idoso_not_death,num_obeso_covid_not_death,num_sp_not_death])

def bairro_analize(all_data_list):

    data_array = np.array(all_data_list)
    id_bairro_covid = np.where(np.logical_and(data_array[:,-3] !='nan',data_array[:,-1] == 'Caso confirmado'))
    bairros_list = [b[-3] for b in data_array[id_bairro_covid[0]]]
    reg1 = [b for b in bairros_list if b == 'Carlito Pamplona' or b =='Barra do Ceará' or b=='Cristo Redentor']
    reg1,labelreg1 = cont_repetido(reg1)
    num_reg1 =  np.sum(labelreg1)

    reg2 = [b for b in bairros_list if b == 'Joaquim Távora' or b == 'Cais do Porto' or b == 'Mucuripe' \
            or b == 'São João do Tauape' or b == 'Vicente Pinzon' or b =='Meireles' or b == 'Papicu']
    reg2,labelreg2 = cont_repetido(reg2)
    num_reg2 =  np.sum(labelreg2)


    reg3 = [b for b in bairros_list if b == 'Parque Araxá' or b == 'Padre Andrade' or b == 'São Gerardo' or b =='Antônio Bezerra']
    reg3,labelreg3 = cont_repetido(reg3)
    num_reg3 =  np.sum(labelreg3)

    reg4 = [b for b in bairros_list if b == 'Jardim América' or b == 'Vila União' or b == 'Montese' or b == 'Vila Peri']
    reg4,labelreg4 = cont_repetido(reg4)
    num_reg4 =  np.sum(labelreg4)

    reg5 = [b for b in bairros_list if b == 'Bonsucesso' or b == 'Bom Jardim' or b=='Granja Lisboa' or b=='Siqueira' or b== 'Granja Portugal']
    reg5,labelreg5 = cont_repetido(reg5)
    num_reg5 =  np.sum(labelreg5)

    reg6 = [b for b in bairros_list if b == 'Lagoa Redonda' or b=='Messejana' or b=='Jardim das Oliveiras'
            or b=='José de Alencar' or b == 'Guajeru' or b == 'Curió']
    reg6,labelreg6 = cont_repetido(reg6)
    num_reg6 =  np.sum(labelreg6)

    reg7 = [b for b in bairros_list if b == 'Castelo Encantado' or b == 'Edson Queiroz' ]
    reg7,labelreg7 = cont_repetido(reg7)
    num_reg7 =  np.sum(labelreg7)

    reg8 = [b for b in bairros_list if b == 'Prefeito José Walter' or b == 'Dias Macedo' or b == 'Passaré' or b == 'Boa Vista' or b=='Serrinha']
    reg8,labelreg8 = cont_repetido(reg8)
    num_reg8 =  np.sum(labelreg8)

    reg9 = [b for b in bairros_list if b == 'Jangurussu' or b =='Ancuri' or b =='Cajazeiras']
    reg9,labelreg9 = cont_repetido(reg9)
    num_reg9 =  np.sum(labelreg9)

    reg10 = [b for b in bairros_list if b == 'Canindezinho' or b == 'Mondubim' or b == 'Manoel Sátiro' or \
             b =='Parque Santa Rosa'or b=='Conjunto Esperança' or b=='Parque São José' or b =='Demócrito Rocha' or b == 'Parque Presidente Vargas']
    reg10,labelreg10 = cont_repetido(reg10)
    num_reg10 =  np.sum(labelreg10)

    reg11 = [b for b in bairros_list if b == 'Genibaú' or b == 'João XXIII' or b == 'Henrique Jorge' \
             or b=='Dom Lustosa' or b == 'Conjunto Ceará I' or b =='Jóquei Clube' or b == 'Pici' or b =='Autran Nunes' or b=='Demócrito Rocha']
    reg11,labelreg11 = cont_repetido(reg11)
    num_reg11 =  np.sum(labelreg11)

    reg12 = [b for b in bairros_list if b == 'Centro']
    reg12,labelreg12 = cont_repetido(reg12)
    num_reg12 =  np.sum(labelreg12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    plot_pizza(ax1,[num_reg1,num_reg2,num_reg3,num_reg4,num_reg5,num_reg6,num_reg7,num_reg8,num_reg9,num_reg10,num_reg11,num_reg12] \
              ,['Regional 1','Regional 2','Regional 3','Regional 4','Regional 5','Regional 6','Regional 7','Regional 8','Regional 9','Regional 10','Regional 11','Regional 12'])
    plot_barh(ax2,[num_reg1,num_reg2,num_reg3,num_reg4,num_reg5,num_reg6,num_reg7,num_reg8,num_reg9,num_reg10,num_reg11,num_reg12],['Regional 1','Regional 2','Regional 3','Regional 4','Regional 5','Regional 6','Regional 7','Regional 8','Regional 9','Regional 10','Regional 11','Regional 12'])


def plot_barh(ax,list_data,list_label):


    ax.barh(list_label,list_data,color='black', )
    ax.set(title="Quantidade de Casos Confirmados de Covid-19 por Regional",xlabel="Quantidade de Casos")

    plt.savefig('src/results/Analise de aquisição de covid por regionais.png', format='png')

def num_of_repetition(value,list):
    return len(np.where(np.array(list)==value)[0])

def cont_repetido(list):
    bairro_list= []
    label_list = []
    for item in list:
     if item not in bairro_list:
        label_list.append(num_of_repetition(item,list))
        bairro_list.append(item)
    return bairro_list,label_list


def plot_pizza(ax1,list_data,list_label):

    return (ax1.pie(list_data, labels = list_label, autopct = '%1.1f%%'))


def data_analize(all_data_list,death_include_list):

    bairro_analize(all_data_list)
    data_analize_covid(all_data_list)
    death_data_analize(death_include_list)


def data_visualize(csv_path):

    all_data_list , death_include_list,_ = data_preprocess(csv_path)
    data_analize(all_data_list,death_include_list)

def data_preprocess(csv_path):
    data_csv = pd.read_csv(csv_path, delimiter=',',encoding='latin-1')
    all_data_list , death_include_list, clf_list = csv_to_list(data_csv)
    return all_data_list,death_include_list,clf_list




