import csv

# Nome do arquivo de entrada e saída
arquivo_entrada = 'connect-4certo.csv'
arquivo_saida = 'connect-4.csv'

# Abrir o arquivo CSV original para leitura
with open(arquivo_entrada, 'r', encoding='utf-8') as entrada:
    leitor = csv.reader(entrada)
    linhas_modificadas = []

    # Ignorar a primeira linha (cabeçalho) e adicionar sem modificações
    cabecalho = next(leitor)
    linhas_modificadas.append(cabecalho)

    # Substituir "b" por "3" nas linhas a partir da segunda
    for linha in leitor:
        linha_modificada = [celula.replace('l2ss', '2') for celula in linha]
        linhas_modificadas.append(linha_modificada)

# Salvar as linhas modificadas em um novo arquivo CSV
with open(arquivo_saida, 'w', encoding='utf-8', newline='') as saida:
    escritor = csv.writer(saida)
    escritor.writerows(linhas_modificadas)

print(f'Arquivo processado e salvo como {arquivo_saida}')
