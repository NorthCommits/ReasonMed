"""
Data pipeline for loading and processing the medical dataset from HuggingFace.
"""

import os
from datasets import load_dataset
import pandas as pd
from typing import List, Dict, Any
import re


class DataPipeline:
    """Handles data loading and processing from HuggingFace dataset."""
    
    def __init__(self, dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT", 
                 subset: str = "en"):
        """
        Initialize the data pipeline.
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset to use (e.g., 'en' or 'en_mix')
        """
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None
    
    def load_dataset(self) -> None:
        """Load the dataset from HuggingFace."""
        print(f"Loading dataset: {self.dataset_name} (subset: {self.subset})...")
        try:
            self.dataset = load_dataset(self.dataset_name, self.subset)
            print(f"Dataset loaded successfully. Splits: {list(self.dataset.keys())}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying without subset parameter...")
            self.dataset = load_dataset(self.dataset_name)
            print(f"Dataset loaded successfully. Splits: {list(self.dataset.keys())}")
    
    def get_train_split(self) -> Any:
        """
        Get the training split from the dataset.
        
        Returns:
            Dataset split (typically 'train')
        """
        if self.dataset is None:
            self.load_dataset()
        
        splits = list(self.dataset.keys())
        if "train" in splits:
            return self.dataset["train"]
        elif len(splits) > 0:
            return self.dataset[splits[0]]
        else:
            raise ValueError("No splits found in dataset")
    
    def extract_keywords(self, text: str) -> str:
        """
        Extract medical keywords from text using simple pattern matching.
        
        Args:
            text: Input text to extract keywords from
            
        Returns:
            Comma-separated string of extracted keywords
        """
        keywords = []
        
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|disorder|condition|infection)\b',
            r'\b(?:fever|pain|headache|nausea|vomiting|diarrhea|cough|shortness of breath|fatigue)\b',
            r'\b(?:CT scan|MRI|X-ray|blood test|urine test|biopsy|ultrasound)\b',
            r'\b(?:medication|treatment|therapy|surgery|prescription)\b',
        ]
        
        text_lower = text.lower()
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend([m.lower() for m in matches])
        
        unique_keywords = list(set(keywords))[:10]
        return ", ".join(unique_keywords) if unique_keywords else "general"
    
    def process_record(self, record: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        Process a single record from the dataset.
        
        Args:
            record: Single record from the dataset
            idx: Index of the record
            
        Returns:
            Processed record with formatted text and metadata
        """
        question = record.get("Question", "")
        reasoning = record.get("Complex_CoT", record.get("Reasoning", ""))
        response = record.get("Response", record.get("Answer", ""))
        
        question_id = f"case_{idx}"
        
        formatted_text = f"Case: {question}\n\nReasoning: {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}\n\nDiagnosis: {response}"
        
        medical_keywords = self.extract_keywords(f"{question} {reasoning} {response}")
        
        return {
            "question_id": question_id,
            "text": formatted_text,
            "full_question": question,
            "full_reasoning": reasoning,
            "full_response": response,
            "medical_keywords": medical_keywords,
        }
    
    def process_all(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Process all records from the dataset.
        
        Args:
            limit: Optional limit on number of records to process
            
        Returns:
            List of processed records
        """
        train_data = self.get_train_split()
        
        processed_records = []
        total = len(train_data) if limit is None else min(limit, len(train_data))
        
        print(f"Processing {total} records...")
        for idx in range(total):
            record = train_data[idx]
            processed = self.process_record(record, idx)
            processed_records.append(processed)
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{total} records...")
        
        print(f"Processing complete. Total records: {len(processed_records)}")
        return processed_records
    
    def to_dataframe(self, processed_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert processed records to pandas DataFrame.
        
        Args:
            processed_records: List of processed records
            
        Returns:
            DataFrame with processed records
        """
        return pd.DataFrame(processed_records)


if __name__ == "__main__":
    pipeline = DataPipeline()
    records = pipeline.process_all(limit=100)
    df = pipeline.to_dataframe(records)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst record:\n{df.iloc[0]}")

