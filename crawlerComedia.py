import os
import requests as req
import regex as re
import numpy as np
from parsel import Selector

TAG_RE = re.compile(r'<[^>]+>')

#Comedia
r = req.get('https://www.imdb.com/search/title/?genres=comedy&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=3396781f-d87f-4fac-8694-c56ce6f490fe&pf_rd_r=WXQQXANVDHDXJAP09K8T&pf_rd_s=center-1&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr1_i_1')
pagina = r.text
#print(pagina)
selector = Selector(text=pagina)

result = selector.xpath('//h3[@class="lister-item-header"]/a/text()').getall()
lista = []
for x in result:
    # tags html
    sentence = TAG_RE.sub('' , x)
    #pontuations
    sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
    # single characteres
    #sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    lista.append(sentence)

listaTitulo = np.array(lista)

result = selector.xpath('//p[@class="text-muted"]').getall()
lista = []
for x in result:
    # tags html
    sentence = TAG_RE.sub('' , x)
    # pontuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # single characteres
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    lista.append(sentence)

listaSinopse = np.array(lista)
print(listaTitulo.shape, listaSinopse.shape, len(listaTitulo))

for i in range(1, 100):
    r = req.get('https://www.imdb.com/search/title/?genres=comedy&start='+str((i*50)+1)+'&explore=title_type,genres&ref_=adv_nxt')
    pagina = r.text

    selector = Selector(text=pagina)

    result = selector.xpath('//h3[@class="lister-item-header"]/a/text()').getall()
    lista = []
    for x in result:
        # tags html
        sentence = TAG_RE.sub('' , x)
        # pontuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # single characteres
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        lista.append(sentence)


    listaTitulo = np.append(listaTitulo, lista)
    result = selector.xpath('//p[@class="text-muted"]').getall()
    lista = []
    for x in result:
        # tags html
        sentence = TAG_RE.sub('' , x)
        # pontuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # single characteres
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        lista.append(sentence)


    listaSinopse = np.append(listaSinopse, lista)

    print(listaTitulo.shape, listaSinopse.shape, len(listaTitulo))


proporcao_treino = 0.5
porporcao_teste1 = 0.5
porporcao_teste2 = 1

fp = open(os.getcwd()+'/database/treino/comedia', 'w')
for i in range(int(len(listaTitulo)*proporcao_treino)):
    #print(listaTitulo[i], ' -', listaSinopse[i], '\n')
    fp.write(listaTitulo[i]+','+'comedia'+','+listaSinopse[i] + '\n')
fp.close()

fp = open(os.getcwd()+'/database/teste/comedia', 'w')
for i in range(int(len(listaTitulo)*porporcao_teste1), int(len(listaTitulo)*porporcao_teste2)):
    #print(listaTitulo[i], ' -', listaSinopse[i], '\n')
    fp.write(listaTitulo[i]+','+'comedia'+','+listaSinopse[i] + '\n')
fp.close()


#fp = open(os.getcwd()+'/database/validacao/comedia', 'w')
#for i in range(int(len(listaTitulo)*0.9), int(len(listaTitulo))):
#    #print(listaTitulo[i], ' -', listaSinopse[i], '\n')
#    fp.write(listaTitulo[i]+','+'comedia'+','+listaSinopse[i] + '\n')
#fp.close()
