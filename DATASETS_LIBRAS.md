# Datasets brasileiros de LIBRAS para treino

Seu dataset atual tinha poucos sinais porque não havia fonte brasileira à mão. Abaixo estão **datasets públicos em LIBRAS** que você pode usar para chegar a **pelo menos 10 palavras** (e bem mais, se quiser).

---

## 1. V-Librasil (recomendado – vocabulário grande)

- **O que é:** Base de vídeos com **1.364 termos em português** em LIBRAS, com **4.089 sinais** (cada termo feito por 3 intérpretes).
- **Onde:**  
  - Site: **https://libras.cin.ufpe.br/**  
  - Download em massa: **IEEE DataPort** – [V-Librasil](https://ieee-dataport.org/documents/v-librasil-new-dataset-signs-brazilian-sign-language-libras)
- **Formato:** Vídeos em MP4 (chroma key), organizados por termo/articulador.
- **Uso:** Pesquisa e educação. **Obrigatório citar** na monografia/trabalho:
  - *Ailton José Rodrigues, "V-LIBRASIL: Uma base de dados com sinais na Língua Brasileira de Sinais (Libras)", Dissertação de Mestrado, Centro de Informática (CIN), Universidade Federal de Pernambuco (UFPE), 2021.*

### Como usar para ter pelo menos 10 palavras

**Opção A – Baixar pelo site (poucas palavras)**  
1. Acesse https://libras.cin.ufpe.br/  
2. Use a **busca** para termos como: **oi**, **obrigado**, **por favor**, **bom dia**, **amor**, **família**, **água**, **comida**, **sim**, **não**, **ajuda**, **desculpa**, **tchau**, **bom**, **feliz**.  
3. Em cada termo, abra a página do sinal e use o link **Download** de cada vídeo.  
4. Crie uma pasta por palavra (ex.: `obrigado/`, `oi/`, `amor/`) e coloque os MP4s dentro.  
5. Aponte o `caminho_videos_originais` do seu `videos.py` para a pasta que contém essas pastas (estrutura: `pasta_raiz/obrigado/video1.mp4`, etc.).

**Opção B – Download em massa (IEEE DataPort)**  
- No IEEE DataPort o dataset está dividido por articulador (arquivos grandes, ~2–5 GB cada).  
- Depois do download, você precisará **organizar os vídeos em pastas por termo** (nome da palavra em português), para ficar igual à estrutura que seu `videos.py` espera: **uma pasta por gesto, com arquivos .mp4 dentro**.

### Sugestão de ≥10 palavras para o treino

Você pode escolher exatamente estas ou outras do V-Librasil (são termos comuns e fáceis de achar na busca):

| # | Palavra   | Observação        |
|---|-----------|-------------------|
| 1 | oi        | Já no seu dataset |
| 2 | obrigado  | Já no seu dataset |
| 3 | de_nada   | Já no seu dataset |
| 4 | desculpa  | Já no seu dataset |
| 5 | eu_amo_voce | Já no seu dataset |
| 6 | comi_muito | Já no seu dataset |
| 7 | por_favor | Buscar no site     |
| 8 | bom_dia   | Buscar no site     |
| 9 | tchau     | Buscar no site     |
|10 | ajuda     | Buscar no site     |
|11 | agua      | Buscar no site     |
|12 | familia   | Buscar no site     |

Assim você mantém suas 6 atuais e adiciona pelo menos mais 4–6 do V-Librasil para passar de 10 palavras.

---

## 2. LIBRAS-HC-RGBDS (UFPR) – configurações de mão

- **O que é:** 61 **configurações de mão** em LIBRAS (não são palavras soltas do dia a dia), 610 vídeos (10 por classe), capturados com Kinect.
- **Onde:**  
  - Página: https://web.inf.ufpr.br/vri/databases/brazilian-sign-language-libras-hand-configurations-database/  
  - Download direto: http://www.inf.ufpr.br/vri/databases/LIBRAS-HC-RGBDS-2011.tar (aprox. 9 GB)
- **Uso:** Pesquisa não comercial, **citando**:  
  - *PORFIRIO, A.; WIGGERS, K.; OLIVEIRA, L. E. S.; WEINGAERTNER, D. "LIBRAS Sign Language Hand Configuration Recognition Based on 3D Meshes". In: 2013 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 2013.*

Serve mais para **complementar** (configurações de mão) do que como vocabulário de "palavras" no sentido de frases do dia a dia. Para "10 palavras" no seu contexto, o **V-Librasil** é mais adequado.

---

## 3. MINDS-Libras (UFMG)

- **O que é:** 20 sinais, cada um gravado 5 vezes por 12 sinalizantes (1.200 amostras no total), com vídeo RGB e profundidade.
- **Onde:** Repositório da UFMG / contato com o laboratório MINDS (dataset descrito em tese da UFMG).
- **Uso:** Boa opção se você conseguir acesso, mas o processo de obtenção é mais burocrático; para **rapidez e vocabulário maior**, o V-Librasil continua melhor.

---

## 4. Libras Movement (UCI)

- **O que é:** 15 classes de **movimento** da mão, 24 instâncias por classe (dados numéricos extraídos dos vídeos, não os vídeos em si).
- **Onde:** https://archive.ics.uci.edu/dataset/181/libras+movement  
- **Uso:** Se você quiser trabalhar com **features numéricas** em vez de vídeo, pode ser útil; para o seu pipeline atual (vídeo → frames → CNN+LSTM), o **V-Librasil** é mais direto.

---

## Integração com seu projeto

Seu `videos.py` espera uma estrutura assim:

```
caminho_videos_originais/
  nome_do_gesto_1/
    video1.mp4
    video2.mp4
  nome_do_gesto_2/
    video1.mp4
  ...
```

- **V-Librasil:** Ao baixar (pelo site ou pelo IEEE DataPort), organize os MP4s em **pastas por termo em português** (ex.: `obrigado/`, `oi/`, `por_favor/`). Use nomes em minúsculas e sem espaços (ex.: `bom_dia`) para bater com o resto do seu código.
- Mantenha o mesmo `caminho_local_temporario` (ex.: `.\imagens_tcc`) para gerar `raw` e `aug_1`, `aug_2` como você já faz.
- Depois rode de novo o treinamento com `cnn_gesture_recognition.py` e o `class_to_idx.json` passará a incluir todas as novas classes.

### Usar vídeos no Google Drive (não pesar a memória local)

**Sim, você pode manter os vídeos no Google Drive** e apontar o `caminho_videos_originais` para a pasta do Drive (ex.: unidade `G:` com Google Drive para Desktop).

- O `videos.py` **só lê** os vídeos do Drive (frame a frame) e **grava** o resultado em `caminho_local_temporario` (ex.: `.\imagens_tcc` no seu PC). Os vídeos originais não são copiados para o disco local.
- O que **ocupa espaço no seu PC** é a pasta **`imagens_tcc`** (todos os frames extraídos + augmentações). O treino depois lê dessa pasta, não dos vídeos.
- **Recomendação:** Deixe os MP4s no Drive; mantenha `imagens_tcc` no disco local para o treino ser mais rápido. Se o disco local ficar apertado, você pode colocar `imagens_tcc` também no Drive, mas o treino tende a ficar mais lento (muitas leituras de arquivos na nuvem).
- Certifique-se de que o Drive está **sincronizado/online** e que o caminho em `videos.py` está correto (ex.: `r"G:\.shortcut-targets-by-id\...\sinais_treinados"` com uma barra invertida entre pastas no Windows).

Se quiser, posso te ajudar a definir uma lista exata de 10–15 palavras no V-Librasil e um passo a passo para deixar a pasta de vídeos pronta para o `videos.py`.

---

## Generalização para a webcam (3 vídeos por palavra)

No V-Librasil há **apenas 3 vídeos por palavra** (um por intérprete). Isso traz dois riscos:

1. **Pouca variedade por classe**  
   Mesmo com `raw` + `aug_1` + `aug_2`, você fica com **9 sequências por palavra** (3 vídeos × 3 versões), todas derivadas dos mesmos 3 intérpretes. O modelo pode decorar características dessas pessoas (tom de pele, roupa, estilo do gesto) em vez do sinal em si.

2. **Diferença de domínio (treino vs teste)**  
   - **Treino:** chroma key, iluminação controlada, 3 intérpretes fixos.  
   - **Teste na webcam:** outra pessoa, fundo qualquer, outra câmera e distância.  
   Essa mudança de domínio tende a **piorar bastante** a precisão na webcam.

### A arquitetura consegue diferenciar na hora do teste?

- **Em teoria, sim:** CNN+LSTM consegue aprender padrões temporais do gesto e generalizar para novos sinalizantes.
- **Na prática, com só 3 vídeos por palavra:** é bem provável que **não** generalize bem para uma pessoa nova na webcam, porque há poucos exemplos e o modelo pode se apoiar em detalhes dos 3 intérpretes ou do fundo.

### O que fazer para melhorar

1. **Incluir você mesma no treino (recomendado)**  
   Grave **as mesmas palavras** na webcam e coloque esses vídeos nas pastas do dataset. Assim o "sinalizante do teste" entra no treino. Mesmo 1–2 vídeos seus por palavra já ajudam muito.

2. **Augmentation em tempo de treino**  
   No `cnn_gesture_recognition.py` foi adicionada a opção `train_augment=True` no `GestureDataset`: a cada época são aplicadas variações aleatórias de cor/brilho/contraste, aumentando a variedade e reduzindo overfitting aos 3 intérpretes.

3. **Mais versões no pré-processamento**  
   No `videos.py` aumente `n_augs` (ex.: de 2 para 4) para gerar mais sequências por vídeo.

4. **Citar a limitação no TCC**  
   Descreva que o dataset tem 3 intérpretes por palavra e que a avaliação na webcam com um sinalizante não visto no treino é uma limitação conhecida.

**Resumo:** Só com os 3 vídeos por palavra do V-Librasil a arquitetura pode até diferenciar as palavras no treino, mas na webcam com outra pessoa o desempenho pode cair. A solução mais efetiva é **misturar dados do V-Librasil com gravações suas** das mesmas palavras e usar **augmentation em tempo de treino** (já disponível no código).
