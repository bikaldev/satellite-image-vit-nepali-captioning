"""
Text preprocessing utilities for Nepali captions.
Handles Devanagari script tokenization and text normalization.
"""

import re
import unicodedata
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class NepaliTextPreprocessor:
    """
    Preprocessor for Nepali text in Devanagari script.
    """
    
    def __init__(self):
        """Initialize Nepali text preprocessor."""
        self.vocab = None
        self.word_freq = None
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Nepali text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization for Nepali text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.normalize_text(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+|\d+', text)
        
        return tokens
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts
            min_freq: Minimum frequency for a word to be included
            
        Returns:
            Vocabulary dictionary (word -> index)
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        # Count frequencies
        self.word_freq = Counter(all_tokens)
        
        # Filter by minimum frequency
        vocab_words = [word for word, freq in self.word_freq.items() if freq >= min_freq]
        
        # Create vocabulary with special tokens
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }
        
        for word in sorted(vocab_words):
            self.vocab[word] = len(self.vocab)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Total unique words: {len(self.word_freq)}")
        
        return self.vocab
    
    def get_caption_statistics(self, captions: List[str]) -> Dict:
        """
        Get statistics about captions.
        
        Args:
            captions: List of captions
            
        Returns:
            Dictionary with statistics
        """
        lengths = [len(self.tokenize(cap)) for cap in captions]
        char_lengths = [len(cap) for cap in captions]
        
        stats = {
            'num_captions': len(captions),
            'avg_word_length': sum(lengths) / len(lengths),
            'max_word_length': max(lengths),
            'min_word_length': min(lengths),
            'avg_char_length': sum(char_lengths) / len(char_lengths),
            'max_char_length': max(char_lengths),
            'min_char_length': min(char_lengths),
            'word_lengths': lengths,
            'char_lengths': char_lengths
        }
        
        return stats
    
    def visualize_statistics(
        self,
        captions: List[str],
        title: str = "Caption Statistics",
        save_path: str = None
    ):
        """
        Visualize caption statistics.
        
        Args:
            captions: List of captions
            title: Plot title
            save_path: Optional path to save plot
        """
        stats = self.get_caption_statistics(captions)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Word length distribution
        axes[0, 0].hist(stats['word_lengths'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(stats['avg_word_length'], color='red', linestyle='--', 
                           label=f"Mean: {stats['avg_word_length']:.1f}")
        axes[0, 0].set_xlabel('Number of Words')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Caption Length Distribution (Words)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Character length distribution
        axes[0, 1].hist(stats['char_lengths'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(stats['avg_char_length'], color='red', linestyle='--',
                           label=f"Mean: {stats['avg_char_length']:.1f}")
        axes[0, 1].set_xlabel('Number of Characters')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Caption Length Distribution (Characters)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Word frequency (top 20)
        if self.word_freq:
            top_words = self.word_freq.most_common(20)
            words, freqs = zip(*top_words)
            
            axes[1, 0].barh(range(len(words)), freqs, color='skyblue', edgecolor='black')
            axes[1, 0].set_yticks(range(len(words)))
            axes[1, 0].set_yticklabels(words, fontsize=9)
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_title('Top 20 Most Frequent Words')
            axes[1, 0].invert_yaxis()
            axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Statistics summary
        summary_text = f"""
        Total Captions: {stats['num_captions']}
        
        Word Length:
        - Average: {stats['avg_word_length']:.2f}
        - Min: {stats['min_word_length']}
        - Max: {stats['max_word_length']}
        
        Character Length:
        - Average: {stats['avg_char_length']:.2f}
        - Min: {stats['min_char_length']}
        - Max: {stats['max_char_length']}
        
        Vocabulary Size: {len(self.vocab) if self.vocab else 'N/A'}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Statistics visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_ngrams(self, captions: List[str], n: int = 2, top_k: int = 20) -> List[tuple]:
        """
        Analyze n-grams in captions.
        
        Args:
            captions: List of captions
            n: N-gram size
            top_k: Number of top n-grams to return
            
        Returns:
            List of (n-gram, frequency) tuples
        """
        from collections import Counter
        
        ngrams = []
        for caption in captions:
            tokens = self.tokenize(caption)
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams.append(ngram)
        
        ngram_freq = Counter(ngrams)
        top_ngrams = ngram_freq.most_common(top_k)
        
        return top_ngrams


def compare_caption_distributions(
    english_captions: List[str],
    nepali_captions: List[str],
    save_path: str = None
):
    """
    Compare English and Nepali caption distributions.
    
    Args:
        english_captions: List of English captions
        nepali_captions: List of Nepali captions
        save_path: Optional path to save plot
    """
    preprocessor_en = NepaliTextPreprocessor()
    preprocessor_np = NepaliTextPreprocessor()
    
    stats_en = preprocessor_en.get_caption_statistics(english_captions)
    stats_np = preprocessor_np.get_caption_statistics(nepali_captions)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Word length comparison
    axes[0].hist(stats_en['word_lengths'], bins=30, alpha=0.6, label='English', edgecolor='black')
    axes[0].hist(stats_np['word_lengths'], bins=30, alpha=0.6, label='Nepali', edgecolor='black')
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Caption Length Distribution (Words)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Character length comparison
    axes[1].hist(stats_en['char_lengths'], bins=30, alpha=0.6, label='English', edgecolor='black')
    axes[1].hist(stats_np['char_lengths'], bins=30, alpha=0.6, label='Nepali', edgecolor='black')
    axes[1].set_xlabel('Number of Characters')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Caption Length Distribution (Characters)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('English vs Nepali Caption Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    
    plt.show()
