import random
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import os.path as osp
import scipy.sparse as sp
from typing import Optional, Callable, List

from sklearn.preprocessing import LabelBinarizer

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import from_networkx,from_scipy_sparse_matrix
from torch_geometric.transforms import Compose

from dataloader.utils import build_relationship, write_to_edgelist


class Pokec(InMemoryDataset):
	url1 = 'http://snap.stanford.edu/data/soc-pokec-profiles.txt.gz'
	url2 = 'http://snap.stanford.edu/data/soc-pokec-relationships.txt.gz'

	def __init__(self, root, region="Bratislava", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):       
		self.name="Pokec"
		self.region=region
		assert self.region in ['Bratislava']
		super().__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])
	
	@property
	def raw_dir(self) -> str:
		return osp.join(self.root, 'raw')

	@property
	def processed_dir(self) -> str:
		return osp.join(self.root, self.region, 'processed')

	@property
	def pokec_processed_dir(self) -> str:
		return osp.join(self.root, 'processed')

	@property
	def raw_file_names(self) -> List[str]:
		return ['soc-pokec-profiles.txt', 'soc-pokec-relationships.txt']

	@property
	def processed_file_names(self) -> str:
		return 'data.pt'

	def download(self):
		download_url(self.url1, self.raw_dir)
		download_url(self.url2, self.raw_dir)
		profiles = f"{self.raw_dir}/soc-pokec-profiles.txt.gz"
		edges = f"{self.raw_dir}/soc-pokec-relationships.txt.gz"
		extract_gz(profiles, self.raw_dir)
		extract_gz(edges, self.raw_dir)

	@property
	def raw_headers(self) -> str:
		headers = "user_id,public,completion_percentage,gender,region,last_login,registration,AGE,body,I_am_working_in_field,spoken_languages,hobbies,I_most_enjoy_good_food,pets,body_type,my_eyesight,eye_color,hair_color,hair_type,completed_level_of_education,favourite_color,relation_to_smoking,relation_to_alcohol,sign_in_zodiac,on_pokec_i_am_looking_for,love_is_for_me,relation_to_casual_sex,my_partner_should_be,marital_status,children,relation_to_children,I_like_movies,I_like_watching_movie,I_like_music,I_mostly_like_listening_to_music,the_idea_of_good_evening,I_like_specialties_from_kitchen,fun,I_am_going_to_concerts,my_active_sports,my_passive_sports,profession,I_like_books,life_style,music,cars,politics,relationships,art_culture,hobbies_interests,science_technologies,computers_internet,education,sport,movies,travelling,health,companies_brands,more"
		return headers.split(",")

	def read_raw_pokec_data(self):
		rows = []
		with open(self.raw_file_names[0], 'r') as fp:
			for line in fp:
				line = line.rstrip()
				row = line.split('\t')
				rows.append(row)
		df = pd.DataFrame.from_records(rows,columns=self.raw_headers)
		print(f"Shape of original df: {df.shape}")
		return df

	# convert `region` text string into parsable county/town/country units
	# see inline comments for rules of parsing 
	def parse_region(self,region):
		# set default values
		county, town, country = np.nan, np.nan, np.nan

		# words captures individual words in the string
		words = region.split(" ")
		# ',' captures units already present in the region string
		units = region.split(",")

		if "kraj," in words:
			county = words[words.index("kraj,")-1]
			# slovakia is never mentioned if kraj is present. its either nothing or it could be cz
			if len(units) == 2:
				if "kraj" in units[0]:
					# combine bratislava - old_town, bratislava - new_town into only bratislava
					# there are some towns with names abefe-frgrv
					# hence the spaces ' - '
					if " - " in units[1]: 
						town_str = units[1].split(" ")
						town = town_str[ town_str.index("-") - 1]
					# otherwise name of town is multiple words long
					else:
						town = units[1][1:] # removes the whitespace at the start
				else:
					town = units[0]
			# by default adds country as slovakia, cz + kraj case handled below
			country = "slovakia"

		# handles the case when kraj is present in region but also country is cz
		if "cz" in words or "ceska republika" in words:
			country = "ceska republika"

		# if zahranicie is present, then only country name is available
		# there are two exceptions of the form zahranicie, bratislava - stare mesto
		# the inner if condition handles these exceptions
		if "zahranicie," in words:
			if "bratislava" in words:
				town = "bratislava"
				country = "slovakia"
			else:
				country = words[words.index("zahranicie")+2]
		
		return county, town, country

	def cleanup_full_raw_data(self):
		df = self.read_raw_pokec_data()
		# parse region to get county, town, and country info
		df['county'], df['town'], df['country'] = zip(*df['region'].map(self.parse_region))
		df = df[df["town"] != " "]
		print(f"Shape of df after creating and cleaning county, town, country: {df.shape}")   
		# drop all rows where region information is absent
		# Note: there are just 163 such rows
		df.dropna(axis=0,subset=["region"], inplace=True)
		df.dropna(axis=0,subset=["gender"], inplace=True)
		df.dropna(axis=0,subset=["AGE"], inplace=True)
     
	def process(self):
		print("In process.")
		if not osp.isfile(osp.join(self.pokec_processed_dir, "pokec_profiles.csv")) and not osp.isfile(osp.join(self.pokec_processed_dir, "pokec.edgelist")):
			self.cleanup_full_raw_data()

		# if self.root/processed doesnt exist:
			# pokec-level process/cleanup
		# if self.root/self.region/raw doesnt exist:
			# trigger region process which will fill region/raw
		# then fill region/processed