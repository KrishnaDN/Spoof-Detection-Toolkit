#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:17:05 2020

@author: krishna
"""

import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, file: str, sr=None, win_len_ms = 0.025, hop_len_ms = 0.010):
        self.file = file
        self.sr = sr
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        if self.sr:
            win_len = self.