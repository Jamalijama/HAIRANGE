# HAIRANGE

This package provides an implementation of the inference pipeline of DataCleaning, Codon2vec, AttentionalPre-trainer,UnsupervisedMachineLearning, ResNetClassifier, AblationNetworks and ReassortmentPredictor.

## Running your DataCleaning

You will need a machine running python environment with any operating systems. 
Full running requires 50GB of disk space to keep genetic databases and intermediate results of the running.

Please follow these steps:

1.  Install the dependencies. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r DataCleaning/requirements.txt
    ```

2.  Make sure that the output directory exists (the default is `DataCleaning/Res`) and that you have sufficient permissions to write into it.

3.  Run `CleaningFile.py` to generate sequence and annotation files from FASTA and gb file.
    The file name parameter in the program is already set, and you can change it as needed. 
    
    Once the run is over, the output directory shall contain the intermediate files from the run and the final result of the run (the default directory is `DataCleaning/Res/final`).
    After generating sequence and annotation files, we generate the strain names and sequences of four fragments into FASTA files (the default directory is `DataCleaning/Res/fasta`) for subsequent sequences alignment respectively.

    ```bash
    python DataCleaning/CleaningFile.py
    ```
    
4.  Use MAFFT v7.520 for sequences alignment of four fragments(the default output directory is `DataCleaning/Res/mafft`).

5.  Run `CleaningAfterMafft.py` to process alignment results and remove redundant sequences.
    The file name parameter in the program is already set, and you can change it as needed. 
    Once the run is over, the output directory shall contain the intermediate files (the default is `DataCleaning/Res/mafft`) from the run and the final result (the default is `DataCleaning/Res/final`) of the run.

    ```bash
    python DataCleaning/CleaningAfterMafft.py
    ```

6.  We make detailed adjustments to the details of the final file, and then we get the final file `new_AIV_all_8_73805.xlsx` (the default directory is `/Res/res1`).

7.  Run `SplitAndShuffle.py` to shuffle and split the final file by year, and then split data of four fragments for subsequent model construction.
    Once the run is over, the output directory (the default is `DataCleaning/Res/res1`) shall contain the intermediate files from the run and the final result of the run.

    ```bash
    python DataCleaning/SplitAndShuffle.py
    ```

## Running your HAIRANGE

1.  Install the dependencies for running HAIRANGE. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.
    
    ```bash
    pip3 install -r requirements.txt
    ```

2.  Make sure that the output directory exists (the default is `Res`) and that you have sufficient permissions to write into it.

### Running your Codon2vec

1.  Run `Batch_codon_count_1.py`, `Batch_codon_count_2.py`, `Batch_codon_count_3.py`, `Batch_codon_count_5.py` to encode the PB2, PB1, PA, and NP fragments into encoding matrix, respectively.
    We encode sequences before 2020 to train single model here, and you can change the `file_name` and `res_name` to encode other sequences.

    ```bash
    python Codon2vec/Batch_codon_count_1.py
    python Codon2vec/Batch_codon_count_2.py
    python Codon2vec/Batch_codon_count_3.py
    python Codon2vec/Batch_codon_count_5.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/npy`) shall contain the encoding matrix of the four fragments.

### Running your AttentionalPre-trainer

NOTE: All '.npy' files under `/Res/npy` was compressed as `/Res/npy.tar.gz`, please uncompress it before utilizing.  

1.  Run `PreTrainning.py` to optimize the  encoding matrix using transformer encoder layer.
    We use sequences before 2020 to train single model here, and you can change the `file_name` to optimize other sequences.
    
    ```bash
    python AttentionalPre-trainer/PreTrainning.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/npy_trm`) shall contain the optimized encoding matrix of the four fragments.

### Running your UnsupervisedMachineLearning

1.  Run `reduce_dim_tocsv_1.py`, `reduce_dim_tocsv_2.py`, `reduce_dim_tocsv_3.py`, `reduce_dim_tocsv_5.py` to reduce dimensionality of the PB2, PB1, PA, and NP embedding matrixes using PCA, tSNE, UMAP methods.
    
    ```bash
    python UnsupervisedML/reduce_dim_tocsv_1.py
    python UnsupervisedML/reduce_dim_tocsv_2.py
    python UnsupervisedML/reduce_dim_tocsv_3.py
    python UnsupervisedML/reduce_dim_tocsv_5.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/trm_feature_csv`) shall contain the reduced dimensionality of the four fragments.

3.  Run `unsupervised_evaluate_1.py`, `unsupervised_evaluate_2.py`, `unsupervised_evaluate_3.py`, `unsupervised_evaluate_5.py` to reduce dimensionality of the PB2, PB1, PA, and NP embedding matrixes using UMAP method and to cluster the dimensionality using AgglomerativeClustering method, then evaluate the cluster results.
    
    ```bash
    python UnsupervisedML/unsupervised_evaluate_1.py
    python UnsupervisedML/unsupervised_evaluate_2.py
    python UnsupervisedML/unsupervised_evaluate_3.py
    python UnsupervisedML/unsupervised_evaluate_5.py
    ```
    
4.  Once the run is over, the output directory (the default is `/Res/pre_processing_evaluate`) shall contain the clustering evaluation results of the four fragments.

### Running your ResNetClassifier

1.  Run `train_single.py` to train single model use encoding matrix before or after transformer according to the results of unsupervised learning.
    
    ```bash
    python ResNetClassifier/train_single.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/fig`) shall contain the result figures of the run.

3.  Run `test_single.py` to test trained single model using the test dataset after 2020.

    ```bash
    python ResNetClassifier/test_single.py
    ```

4.  Once the run is over, the output directory (the default is `/Res/prediction_result_csv`) shall contain the results of the run.

### Running your AblationNetworks

1.  Run `ablation_single.py` to execute ablation experiments.
    
    ```bash
    python AblationNetworks/ablation_single.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/ablation_result`) shall contain the results of the run.

### Running your ReassortmentPredictor

1.  Run `reassort_npy.py` to concat encoding vectors of all the codons of four fragments together.
    We filled with zero vectors to satisfy the required shape for training the model.
    We concated sequences with all human samples before 2020, 95% avian H1N1 subtype and 5% avian H3N2 subtype samples before 2020 here, you can change the `file_name` to concat other sequences.
    
    ```bash
    python ReassortmentPredictor/reassort_npy.py
    ```
    
2.  Once the run is over, the output directory (the default is `/Res/npy_reassort`) shall contain the results of the run.
    
3.  Run `train_reassort.py` to train the reassortment model.
    
    ```bash
    python ReassortmentPredictor/train_reassort.py
    ```
    
4.  Once the run is over, the output directory (the default is `/Res/fig`) shall contain the result figures of the run.
    
5.  Run `test_reassort.py` to predict the test dataset and given reassortment dataset using the trained reassortment model.
    We use given reassort sequences here, you can change the `file_name` to test other sequences.

    ```bash
    python ReassortmentPredictor/test_reassort.py
    ```

6.  Once the run is over, the output directory (the default is `/Res/prediction_result_csv`) shall contain the results of the run.

7.  Run `reassort_single.py` to concat encoding vectors of all the codons without filling zero vectors of four fragments respectively and generate four output files.
    
    ```bash
    python ReassortmentPredictor/reassort_single.py
    ```

8.  Once the run is over, the output directory (the default is `/Res/npy_reassort`) shall contain the results of the run.

9.  Run `simulate_seq.py` to generate simulated reassortment sequences in encoding matrix format with two model sequences of human H3N2 subtype.
    
    ```bash
    python ReassortmentPredictor/simulate_seq.py
    ```
    
10. Once the run is over, the output directory (the default is `/Res/npy_reassort`) shall contain the results of the run.

11. Run `test_simulated.py` to predict simulated reassortment sequences.

    ```bash
    python ReassortmentPredictor/test_simulated.py
    ```
    
12. Once the run is over, the output directory (the default is `/Res/prediction_result_csv`) shall contain the results of the run.

    
NOTE: THE ORIGINAL DATA FILEs IN "/DataCleaning/Res/res1", SOME IMPORTANCE FILEs FOR TRAINING THE MODELS 
AND SOME RESULT FILEs IN "/Res" HAVE BEEN COMPRESSED INTO SOME FILE "HAIRANGE-XXX.zip".
AND THE ZIP FILE HAS BEEN UPLOADED ON THE WEBSITE OF "10.5281/zenodo.12747426". 
PLEASE DOWNLOAD THE ZIP FILE AND DECOMPRESS IT UNDER THE HOME DIRECTORY OF 'HAIRANGE'.
