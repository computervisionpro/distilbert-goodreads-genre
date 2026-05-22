
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class GoodreadsGenrePredictor:
    def __init__(self, model_repo):
        """
        Initializes the inference pipeline by detecting hardware constraints,
        and downloading the model weights along with the fast tokenizer configuration.
        """

        self.model_repo = model_repo
        
        # 1. Hardware Context Identification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing inference session utilizing device: {self.device}")
        
        # 2. Strict ID-to-Label Class Mapping Definition
        
        self.id2label = { 
                0: "Young Adult (YA)",
                1: "Comics & Graphic Novels",
                2: "History & Biography",
                3: "Mystery, Thriller, & Crime",
                4: "Fantasy & Paranormal",
                5: "Poetry",
                6: "Romance",
                7: "Children's Literature"

                }
        
        # 3. Component Retrieval and Serialization Setup
        print(f"[INFO] Fetching artifacts from repository: '{self.model_repo}'...")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_repo)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_repo, use_safetensors=True)

        # Bind model parameters explicitly to target hardware context and lock to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] Model loaded successfully.")

    def predict(self, text: str):
        """
        Processes an individual raw text document string, transforms it into context 
        tensors, feeds it into the model matrix, and returns prediction details.
        """
        # Ensure sequence boundaries match the fine-tuning training configurations
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Map input dictionaries directly into hardware-assigned memory structures
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Disable gradient calculations to maximize forward pass processing speed
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract classification logit distributions
        logits = outputs.logits
        
        # Apply Softmax layer sequence to derive continuous confidence scores
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Extract index position of highest likelihood probability element
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence_score = probabilities[0][predicted_class_id].item()
        
        predicted_label = self.id2label.get(predicted_class_id, "Unknown Genre Structure")
        
        return {
            "genre_label": predicted_label,
            "class_id": predicted_class_id,
            "confidence": confidence_score
        }

# =====================================================================
if __name__ == "__main__":
    # Initialize the prediction class
    predictor = GoodreadsGenrePredictor(model_repo="computervisionpro/distilbert-goodreads-genres")
    
    # Example review strings mimicking different literary categories 
    sample_reviews = [
        # Example 1: Dark Mystery/Thriller theme
        "The detective followed the trail of breadcrumbs deep into the city's alleyways. "
        "Every suspect had a rock-solid alibi, but the blood-stained note found at the crime scene "
        "pointed to a sinister conspiracy that reached the highest levels of the government. A gripping thriller!",
        
        # Example 2: Classic Epic Fantasy/Paranormal theme
        "An ancient dark lord awakens in the northern wastes, rallying legions of orcs and dragons. "
        "Only the chosen mage, armed with the legendary crystal staff and an alliance of elves, can close "
        "the dimensional portal before the kingdom falls into eternal darkness.",
        
        # Example 3: Deep History/Biography theme
        "This brilliant biography offers an unparalleled deep dive into the political maneuvers and military "
        "strategies of the Roman Empire during the second century. The author meticulously leverages thousands "
        "of ancient letters, translated archives, and archaeological records to paint a vivid picture of the Emperor."
    ]
    
    print("\n\n")
    print("RUNNING INFERENCE TESTS")
    print("="*70)
    
    for idx, review in enumerate(sample_reviews, start=1):
        print(f"\n[Test Case #{idx}]:")
        # Display an abbreviated preview snippet of long review text inputs
        print(f"Truncated Content: \"{review[:120]}...\"")
        
        # Fire prediction pipeline calculation pass
        result = predictor.predict(review)
        
        print(f"  --> Predicted Category : {result['genre_label']}")
        print(f"  --> Structural Model ID: {result['class_id']}")
        print(f"  --> Pipeline Confidence: {result['confidence']:.2%}")
        print("-" * 50)


