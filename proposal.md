# Multidisciplinary Project — Alessandro

## Workload-Driven Exploration of Edge NPU Architectures for Vision-Language-Action Models

## Obiettivo

I **Vision-Language-Action (VLA) models** sono una classe emergente di modelli per robotica ed embodied AI che combinano percezione visiva, comprensione del linguaggio e generazione di azioni.

Dal punto di vista architetturale, un VLA non è un workload uniforme: vision encoder, backbone VLM/Transformer e action generation possono avere caratteristiche molto diverse in termini di dimensioni delle matrici, parallelismo, riuso dei dati e richiesta di banda di memoria.

L'obiettivo del progetto è capire **come VLA rappresentativi si mappano su una NPU edge** e verificare se una singola organizzazione hardware omogenea sia adatta a tutte le fasi del workload oppure se esista una motivazione concreta per architetture **eterogenee o riconfigurabili**.

Il progetto non parte quindi da una nuova architettura già decisa. Si parte dal workload, si individuano i bottleneck e solo successivamente si valuta quale organizzazione hardware potrebbe essere più adatta.

---

## Domande principali

Il progetto dovrebbe rispondere principalmente a queste domande:

1. **Come è composto computazionalmente un VLA moderno?**  
   Quali sono le principali fasi, quali operatori dominano, quali GEMM/GEMV vengono eseguite e con quali dimensioni \(M,N,K\)?

2. **Le diverse fasi del VLA hanno esigenze architetturali differenti?**  
   Ad esempio: alcune preferiscono grandi systolic array mentre altre soffrono di bassa utilization? Alcune sono compute-bound e altre memory-bound?

3. **Quanto cambiano queste caratteristiche tra diverse famiglie di VLA?**  
   L'obiettivo è capire se esiste un comportamento comune oppure se modelli con action generation differente richiedono hardware differente.

4. **Una singola NPU omogenea rimane vicina all'ottimo per tutto il workload?**  
   Oppure configurazioni differenti — dimensione/numero degli array, memoria on-chip, bandwidth — risultano ottimali per fasi differenti?

La risposta a quest'ultima domanda dovrà fornire la motivazione principale per una possibile tesi successiva su una NPU **eterogenea, specializzata o riconfigurabile**.

---

## Approccio sperimentale

L'idea è utilizzare strumenti esistenti, evitando di costruire un nuovo simulatore.

### 1. Workload reale: LeRobot / PyTorch

Usare implementazioni open-source di VLA reali, inizialmente **SmolVLA** e successivamente almeno un secondo modello, ad esempio **π0/π0.5**.

L'esecuzione PyTorch viene usata come riferimento per ricavare:

- suddivisione nelle principali fasi;
- operatori eseguiti;
- shape dei tensori;
- dimensioni delle GEMM \(M,N,K\);
- quantità di calcolo e dati movimentati.

Il profiling dovrebbe essere basato il più possibile sugli strumenti standard di PyTorch (`torch.profiler`). Non serve creare un formato complesso: è sufficiente mantenere il trace originale e produrre una semplice tabella delle operazioni utile per l'analisi.

### 2. Design-space exploration: SCALE-Sim v3

Le GEMM rilevanti vengono convertite nel normale formato \(M,N,K\) utilizzato da **SCALE-Sim v3**.

SCALE-Sim sarà il principale strumento per esplorare rapidamente diverse organizzazioni di NPU, ad esempio variando:

- dimensione e numero dei systolic array;
- dataflow;
- memoria on-chip;
- bandwidth esterna.

Le metriche principali saranno latency/cycles, utilization, stalls e traffico verso SRAM/DRAM.

### 3. Verifica più dettagliata: PyTorchSim

**PyTorchSim** può essere utilizzato successivamente su alcuni blocchi/configurazioni rappresentativi per verificare che le tendenze osservate con SCALE-Sim rimangano valide includendo una modellazione più completa della NPU (vector unit, DMA, memoria, NoC, compiler mapping).

PyTorchSim è quindi uno strumento di **validazione selettiva**, non una dipendenza necessaria per far funzionare il progetto.

In sintesi:

```text
VLA reale in PyTorch
        ↓
profiling / M,N,K / memoria
        ↓
SCALE-Sim → esplorazione architetturale
        ↓
PyTorchSim → verifica di alcuni casi interessanti
```

---

## Modelli da considerare

Per iniziare:

- **SmolVLA** — modello relativamente piccolo e pensato anche per deployment efficienti;
- **π0 / π0.5** — famiglia VLA molto rilevante e architetturalmente più complessa.

Un terzo modello può essere aggiunto successivamente se utile, soprattutto per verificare se le conclusioni ottenute generalizzano.

---

# Materiale da studiare

L'obiettivo non è leggere tutto prima di iniziare. Lo studio dovrebbe procedere insieme agli esperimenti.

## 1. Fondamenti di acceleratori DNN

### Sze et al. — *Efficient Processing of Deep Neural Networks: A Tutorial and Survey*
https://arxiv.org/abs/1703.09039

Da capire soprattutto:
- perché il data movement è così importante;
- reuse e memory hierarchy;
- dataflow;
- metriche utilizzate per valutare un acceleratore.

### Eyeriss — *An Energy-Efficient Reconfigurable Accelerator for Deep CNNs*
https://doi.org/10.1109/JSSC.2016.2616357

Focus: PE organization, dataflow, reuse e relazione tra workload e architettura.

### TPU — *In-Datacenter Performance Analysis of a Tensor Processing Unit*
https://arxiv.org/abs/1704.04760

Focus: systolic array, utilization, bandwidth e differenza tra peak performance e performance realmente ottenibile.

---

## 2. Transformer e Vision Transformer

### *Attention Is All You Need*
https://arxiv.org/abs/1706.03762

Non serve approfondire il training. È importante capire la struttura computazionale di attention e MLP.

### *An Image is Worth 16x16 Words — Vision Transformer*
https://arxiv.org/abs/2010.11929

Focus: come la parte vision viene trasformata in un workload Transformer.

---

## 3. Vision-Language-Action models

### π0 — *A Vision-Language-Action Flow Model for General Robot Control*
https://arxiv.org/abs/2410.24164

Focus: struttura del modello, VLM backbone, action expert e flow matching.

### SmolVLA
Paper: https://arxiv.org/abs/2506.01844  
LeRobot: https://huggingface.co/docs/lerobot/main/smolvla

Questo sarà probabilmente il primo workload da eseguire e profilare.

---

## 4. Stato dell'arte sul problema architetturale

### *Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures* (2026)
https://arxiv.org/abs/2603.02271

### *Characterizing Vision-Language-Action Models across Heterogeneous Edge Accelerators* (2026)
https://arxiv.org/abs/2604.24447

Questi lavori sono particolarmente importanti perché mostrano che l'inferenza VLA presenta bottleneck architetturali specifici. Devono essere letti criticamente per capire **cosa è già noto e quali domande rimangono aperte**.

---

## 5. Strumenti di simulazione

### SCALE-Sim v3
https://github.com/scalesim-project/scale-sim-v3

Capire come descrivere un workload GEMM e come cambiano cycles, utilization e memory traffic modificando la configurazione dell'acceleratore.

### PyTorchSim
https://github.com/PSAL-POSTECH/PyTorchSim

Da studiare dopo SCALE-Sim. L'obiettivo è capire quale livello di dettaglio aggiunge e se può essere utilizzato per validare alcuni blocchi VLA rappresentativi.

---

## Primi passi

Per iniziare il progetto:

1. studiare i concetti fondamentali di DNN accelerator, systolic array e Transformer;
2. installare e provare **SCALE-Sim** su alcuni semplici GEMM, osservando come cambia la utilization al variare dell'array;
3. studiare la struttura di **SmolVLA** e riuscire ad eseguire una inference in LeRobot/PyTorch;
4. iniziare a identificare e profilare le principali fasi e operazioni del modello.

Il passo successivo sarà trasformare questa caratterizzazione in una vera esplorazione dello spazio architetturale della NPU.

---

## Possibile prosecuzione in tesi

Se il progetto mostrerà che differenti fasi dei VLA richiedono configurazioni hardware sensibilmente diverse, la tesi potrà studiare una **NPU eterogenea o riconfigurabile per VLA inference**, progettata a partire dai bottleneck osservati durante il Multidisciplinary Project.

La struttura precisa dell'architettura non viene definita ora: dovrà emergere dai risultati del progetto.
