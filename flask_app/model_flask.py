import wtforms as wtf 
import numpy as np

class Figure(wtf.Form):
	filename = wtf.FileField(validators=[wtf.validators.InputRequired()])
	thresh = wtf.FloatField(label="Threshold")
	show_match = wtf.BooleanField(label="Show suspicious images", default=True)
	#figure_idx = wtf.SelectMultipleField(label="Figure Index", choices = [('1', '2'), ('a', 'b'), ('un', 'pg')])

class Match(wtf.Form):
	group_id = wtf.SelectField(label="Choose match numbers", coerce=str)
	affine_match = wtf.BooleanField(label= "Orientation and Scale", default=True)
	contrast = wtf.BooleanField(label="Contrast", default=True)
	heatmap = wtf.SelectField(label="Display type", choices=[("gray", "Normal"), ("hot", "Heat")], coerce=str)




