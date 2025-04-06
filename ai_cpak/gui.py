"""
Graphical user interface for running the model
Author: Jack Bosco
breezypythongui Author: Ken Lambert
GUI Documentation: https://lambertk.academic.wlu.edu/breezypythongui/
"""

from tkinter import PhotoImage
from .utils.breezypythongui import EasyFrame
import pickle
import pandas as pd
import config

class NYUMethodGUI(EasyFrame):
	"""Gives model MPTA and LDFA predictions after
	normalizing input MPTA and LDFA"""

	def __init__(self):
		EasyFrame.__init__(self, title="AI CPAK Tool")
		self.master.wm_iconphoto(True, PhotoImage(file='assets/logo.png', format='PNG'))

		# Label and field for inputs
		self.addLabel(text = "Pre-op MPTA",
				row=0,column=0)
		self.mpta_in = self.addFloatField(value=None,
									   row=0,
									   column=1,
									   width=20)
		self.addLabel(text = "Pre-op LDFA",
				row=1,column=0)
		self.ldfa_in = self.addFloatField(value=None,
									   row=1,
									   column=1,
									   width=20)

		#command button
		self.addButton(text="Predict", row=2, column=0,
				 columnspan=2, command = self.predict)

		#--- outputs -----
		# output ldfa
		self.addLabel(text="Predicted MPTA",
									row=4,
									column=0)
		self.out_mpta = self.addFloatField(value=None,
									 row=4,
									 column=1,
									 width=8,
									 precision=2,
									 state="readonly")

		self.addLabel(text="Predicted LDFA",
									row=5,
									column=0)
		self.out_ldfa = self.addFloatField(value=None,
									 row=5,
									 column=1,
									 width=8,
									 precision=2,
									 state="readonly")


		# data normalizer
		self.normalizer=pickle.load(open(config.norm_path, 'rb'))
		self.denormalizer=pickle.load(open(config.de_norm_path, 'rb'))

		# pre-trained model
		self.model=pickle.load(open(config.model_path, 'rb'))

		# dataframe for compatability with scikit-learn
		self.data=pd.DataFrame(data={'Pre-op mpta':[], 'Pre-op ldfa':[]}, dtype=float)

	def predict(self):
		mpta = self.mpta_in.getNumber()
		ldfa = self.ldfa_in.getNumber()

		self.data['Pre-op mpta'] = [mpta]
		self.data['Pre-op ldfa'] = [ldfa]

		inpts = self.normalizer.transform(self.data)
		preds=self.model.predict(inpts)
		preds=self.denormalizer.inverse_transform(preds)
		o_mpta, o_ldfa = preds[-1]
		self.out_ldfa.setNumber(o_ldfa)
		self.out_mpta.setNumber(o_mpta)

if __name__ == "__main__":
	NYUMethodGUI().mainloop()
