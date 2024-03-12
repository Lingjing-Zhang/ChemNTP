import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.SaltRemover import SaltRemover
# from molvs.normalize import Normalizer, Normalization
from molvs import Standardizer
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Reionizer, Uncharger
from molvs.tautomer import TAUTOMER_TRANSFORMS, TAUTOMER_SCORES, MAX_TAUTOMERS, TautomerCanonicalizer, TautomerEnumerator, TautomerTransform
import pandas as pd



def Standardize(x):
	testsmi = str(x)
	remover=SaltRemover()
	try:
		mol = Chem.MolFromSmiles(testsmi)
		# mol = Normalizer(mol)
		mol = Standardizer().standardize(mol)
		mol = LargestFragmentChooser()(mol)
		mol = Uncharger()(mol)
		mol = TautomerCanonicalizer().canonicalize(mol)
		mol2=remover.StripMol(mol)
		smiles = Chem.MolToSmiles(mol2)
		if mol2.GetNumAtoms()==0:
			return "None"
		else:
			return smiles
	except:
		return "None"

smi=['C1=CC(=C(C=C1Br)Br)OC2=C(C(=C(C=C2)Br)O)Br','COC1=C(C=CC(=C1Br)OC2=C(C=C(C=C2)Br)Br)Br','C1=CC(=C(C=C1Br)Br)OC2=C(C=C(C(=C2)O)Br)Br','C1=CC(=C(C=C1Br)Br)OC2=C(C=C(C=C2Br)Br)O','COC1=C(C(=CC(=C1)Br)Br)OC2=C(C=C(C=C2)Br)Br','C1=C(C(=CC(=C1OC2=CC(=C(C=C2Br)Br)Br)Br)Br)O','C1=C(C=C(C(=C1O)OC2=CC(=C(C=C2Br)Br)Br)Br)Br','COC1=C(C(=CC(=C1)Br)Br)OC2=CC(=C(C=C2Br)Br)Br']
for i in smi:
	x=Standardize(i)
	print(x)


