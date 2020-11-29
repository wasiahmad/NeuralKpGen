#!/usr/bin/env python3
import logging
from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO, filename="log.txt")
model = BertModel.from_pretrained("./download/")
model.save_pretrained("./")

tok = BertTokenizer.from_pretrained("./download/")
tok.save_pretrained("./")
