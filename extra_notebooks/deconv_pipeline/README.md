# nuSVR deconvolution of cell-free RNA using Tabula Sapiens v 1.0

To deconvolve samples:
Step 1. Conda environment
	- Create the conda environment with the packages using 'deconv_env.yml'
	- Note: if operating on a non-linux system and you experience package dependency issues, you can create a conda environment with the following packages
		(1) python 3.7.4
		(2) scikit-learn 0.24.2
		(3) numpy 1.18.1
		(4) cvx-opt 1.2.5
		(5) scipy 1.5.2

Step 2. Basis Matrix
	- Tabula Sapiens v1 basis matrix (as in this work): unzip the basis matrix (`gunzip tsp_v1_basisMatrix.txt.gz`)
	- Custom basis matrix: make sure that you have the valid file path.
		* Required format: first column is a list of genes and each subsequent column corresponds to a sample 
	- 🚨 IMPORTANT NOTES
		* If using a custom basis matrix, update the header of `deconvolve.py` with its path
		* The units must match between the basis matrix two (e.g. both CPM or both TPM, etc) and **there must be no log-transformation**. If you're using the TSP v1 basis matrix, the units are CPM. 


Step 3. Bulk RNA samples for deconvolution
	- 🚨 IMPORTANT NOTES
		- If you are using the TSP v1 basis matrix, your samples *must* be CPM-normalized. If you're working with cfRNA, normalize for plasma volume too.
		- If you are using your own basis matrix, the normalization units of the samples must correspond to the normalization scheme of the basis matrix. 
		- Ensure that your sample file path corresponds to a file where the first column is a list of genes and each subsequent column corresponds to a single sample. Check out the sample sheet need be.
		- The sample names cannot contain any '-' character. Please switch to "_" or some other character.
		- The genes must have the same naming convention as that of the basis matrix. If you're using the TSP v1 basis matrix, use gene names (e.g. "NRGN" etc). Ensembl ID or Entrez gene ID will not work.
 
Step 4. Update sample_names.txt for your own samples 

Step 5. Launch Jobs!
	- Launch all sample deconvolution jobs simultaneously using the run_all_samples.sh (sbatch run_all_samples.sh)

Step 6. When your jobs complete, run `merge_nnls.py`, this will write out two files, one of all the combined coefficients and one of all the support vectors corresponding to a given nu/C combination for a  given sample.
 
PLEASE READ: couple deconvolution notes:
- Do *not* pass in samples in log-transformed space. For a full proof for why this is inappropriate, please check out this reference: (Zhong, Y. & Liu, Z., Nature Methods 2012)
- The fractions that come out of this program using this program denote the relative fractional contributions of cell type specific RNA for the 23 tissues from which these cell types originate (if using the Tabula Sapiens v1.0 basis matrix) 
- If you are looking for signal from a specific tissue-specific cell type in cfRNA, it is advised to perform signature scoring in conjunction with systems-level deconvolution.
