# Progetto_FVAB
Questo repository contiene il codice e le istruzioni necessarie per eseguire il progetto relativo al corso di Fondamenti di Visione Artificiale e Biometria. 

## Struttura del progetto

| Cartella / File                                     | Descrizione                                                                 |
|-----------------------------------------------------|-----------------------------------------------------------------------------|
| `archive/`                                          | Dataset. Il progetto richiede un dataset specifico che non è incluso nel repository a causa delle sue dimensioni. Il dataset è presente al seguente repository: https://github.com/awsaf49/artifact        |
| `Risultati_SVM_2/`                                        | Contiene i risultati relativi alle coppie di modelli                           |
| `Risultati_SVM_4/`                                  | Contiene i risultati relativi agli esperimenti che coinvolgevano 4 modelli                                |
| `RisultatiSiameseDiversi/`                          | Risultati del modello Siamese su immagini di tipo "diverso" (mix di dati reali e Deepfake)               |
| `RisultatiSiameseSimili/`                           | Risultati del modello Siamese su immagini di tipo "simile" (4 modelli Deepfake)                 |
| `.gitignore`                                        | File di configurazione per Git   |
| `environment.yml`                                   | File per creare l'ambiente Conda con tutte le dipendenze del progetto      |
| `README.md`                                         | README del progetto                                     |
| `SVM_2.ipynb`                                       | Notebook per l’esperimento SVM 2 (coppie di modelli)                                           |
| `SVM_4_fake.ipynb`                                  | Notebook per l’esperimento SVM 4 (4 modelli Deepfake)                            |
| `SVM_4_real_fake.ipynb`                             | Notebook per esperimento SVM 4 con mix di dati reali e Deepfake          |
| `test_artifact_siamese_vit_fake.py`                | Script di test del modello Siamese (4 modelli Deepfake)                      |
| `test_artifact_siamese_vit_real_fake.py`           | Script di test del modello Siamese su un mix di dati reali e Deepfake                |
| `train_artifact_siamese_vit_fake.py`               | Script di training del modello Siamese (4 modelli Deepfake)                 |
| `train_artifact_siamese_vit_real_fake.py`          | Script di training del modello Siamese su un mix di dati reali e Deepfake           |
| `versionitorch.py`                                  | Script per il logging della versione di PyTorch e configurazioni varie     |

## Istruzioni per l'installazione
### Creazione di un ambiente virtuale con conda e installazione delle dipendenze

```bash
conda env create -f environment.yml
```

## Documentazione
Link alla documentazione ufficiale del progetto: TO ADD
