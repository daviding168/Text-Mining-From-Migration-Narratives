# coding = utf-8
import numpy as np
import os
from ..utils import read_patterns_file
import pandas as pd


class Pattern():
    """
    Class that enables to store pattern and match with pattern found in real data.
    """

    def __init__(self, language, freq_pattern_min=5):
        """
        Constructor
        Parameters
        ----------
        language : str
            pattern language
        freq_pattern_min : int
            minimum frequency of the patterns used
        """
        self.language = language
        # dataframe of a pattern file
        self.df_patt = read_patterns_file(language)
        basedir = os.path.dirname(__file__)
        # access pattern file
        filename = os.path.join(basedir, "../resources/treetagger_spacy_mappings/{0}.csv".format(language))
        # mapping two columns of a csv file to create dictionary
        tt_spacy = dict(pd.read_csv(filename, sep="\t", header=None).values)
        # mapping spacy pos with TreeTagger pos (convert and get the value from dictionary tt_spacy)
        self.df_patt["pattern"] = self.df_patt.pattern.apply(lambda x: " ".join([tt_spacy[i] for i in x.split(" ")]))
        self.df_patt = self.df_patt[self.df_patt.frequency > freq_pattern_min]
        # list of patterns corresponds to number of pattern used before delete the redundant
        self.patterns = self.df_patt.pattern.values
        # list of frequencies value
        self.frequencies = self.df_patt.frequency.values
        # list of patterns corresponds to number of pattern used after deleted the redundant
        self.a = np.arange(len(self.patterns))

    def delete_pattern_dupes(self):
        """
        Group patterns and their frequencies if their patterns became the same
        after the "Treetagger -> spacy" operation
        """
        unique_patterns = np.unique(self.patterns)
        sum_unique_frequencies = np.empty(len(unique_patterns)).astype(int)
        for pat_num in range(len(unique_patterns)):
            dupes_list = np.flatnonzero(np.asarray(self.patterns == unique_patterns[pat_num]))
            new_frequency = 0
            for dupe in dupes_list:
                new_frequency = new_frequency + self.frequencies[dupe]
            sum_unique_frequencies[pat_num] = new_frequency
        self.patterns = unique_patterns
        self.frequencies = sum_unique_frequencies

    def get_longest_pattern(self):
        """
        Return the length of the longest pattern available in Patter.patterns

        Returns
        -------
        int
        """
        word_count = np.vectorize(lambda x: len(x.split(" ")))
        return np.max(word_count(self.patterns))

    def match_slow(self, pos_tags_sequence):
        matched = self.df_patt[self.df_patt.pattern == " ".join(pos_tags_sequence)].copy()
        if len(matched) > 0:
            return True, matched.iloc[0].pattern, matched.iloc[0].frequency
        return False, "", 0

    def match(self, pos_tags_sequence):
        """
        Check if a pattern found in real data exists in our pattern database
        Parameters
        ----------
        pos_tags_sequence : list
            pattern found

        Returns
        -------
        bool,str,int
            if the pattern exists, matched pattern sequence, matched pattern frequency
        """

        # index of list of all patterns when matching with pos_tags_sequence
        index = self.a[self.patterns == " ".join(pos_tags_sequence)]
        if len(index) > 0:
            return True, self.patterns[index[0]], self.frequencies[index[0]]

        return False, "", 0

    def sum_all_patterns_frquency(self):
        """
        Sum of all patterns' frequency in our database.
        Returns
        -------
        int
        """
        return self.df_patt.frequency.sum()
