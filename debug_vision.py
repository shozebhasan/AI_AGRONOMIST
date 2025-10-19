"""
debug_vision.py - Standalone script to test vision models

Usage:
    python debug_vision.py path/to/image.jpg
    python debug_vision.py path/to/image.jpg --crop cotton
"""

import sys
import os
import base64
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your vision module
from vision import (
    MODELS, 
    META, 
    decode_base64_to_pil, 
    preprocess_for_model,
    _predict_generic,
    predict_crop_from_base64,
    analyze_image_auto,
    predict_best_model_from_base64
)


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def test_single_model(model_name: str, b64_image: str):
    """Test a single model and show detailed results"""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    try:
        # Get prediction
        result = _predict_generic(model_name, b64_image)
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"  Crop:        {result['crop']}")
        print(f"  Status:      {result['status']}")
        print(f"  Label:       {result['label']}")
        print(f"  Confidence:  {result['confidence']:.2%}")
        print(f"  Description: {result['description']}")
        
        # Get all class probabilities
        model = MODELS[model_name]
        meta = META[model_name]
        pil_img = decode_base64_to_pil(b64_image)
        tensor = preprocess_for_model(pil_img, target_size=meta["target_size"], channels=meta["channels"])
        preds = model.predict(tensor, verbose=0)[0]
        
        print(f"\n📈 ALL CLASS PROBABILITIES (sorted by confidence):")
        print(f"  {'Rank':<6} {'Class Name':<30} {'Probability':<12} {'Percentage':<12} {'Status'}")
        print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
        
        # Sort predictions by probability
        sorted_indices = sorted(range(len(preds)), key=lambda i: preds[i], reverse=True)
        
        for rank, idx in enumerate(sorted_indices, 1):
            prob = preds[idx]
            label = meta["labels"].get(idx, f"class_{idx}")
            marker = "⭐ PREDICTED" if idx == result.get('index', -1) else ""
            print(f"  {rank:<6} {label:<30} {prob:<12.6f} {prob*100:<11.2f}% {marker}")
        
        # Additional metrics
        print(f"\n📐 PREDICTION METRICS:")
        sorted_probs = sorted(preds, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        print(f"  Confidence Gap (1st - 2nd):  {gap:.4f} ({gap*100:.2f}%)")
        print(f"  Top-2 Ratio (1st / 2nd):     {sorted_probs[0] / sorted_probs[1] if len(sorted_probs) > 1 and sorted_probs[1] > 0 else 'N/A'}")
        print(f"  Input Shape:                 {meta['target_size']} x {meta['channels']} channels")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_all_models(b64_image: str):
    """Test all models and compare results"""
    print(f"\n{'#'*80}")
    print(f"# TESTING ALL MODELS")
    print(f"{'#'*80}")
    
    results = {}
    
    for model_name in ["cotton", "wheat", "corn", "rice"]:
        result = test_single_model(model_name, b64_image)
        if result:
            results[model_name] = result
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY: Model Comparison")
    print(f"{'='*80}")
    print(f"  {'Model':<10} {'Predicted Label':<30} {'Confidence':<12} {'Status'}")
    print(f"  {'-'*10} {'-'*30} {'-'*12} {'-'*10}")
    
    # Sort by confidence
    sorted_results = sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    for model_name, result in sorted_results:
        marker = "🏆" if result == sorted_results[0][1] else "  "
        print(f"{marker} {model_name.capitalize():<10} {result['label']:<30} {result['confidence']:<12.2%} {result['status']}")
    
    return results


def test_auto_detection(b64_image: str, threshold: float = 0.50):
    """Test auto-detection algorithm"""
    print(f"\n{'='*80}")
    print(f"AUTO-DETECTION TEST (threshold={threshold})")
    print(f"{'='*80}")
    
    try:
        # Run detailed prediction
        detailed_result = predict_best_model_from_base64(b64_image, require_threshold=threshold)
        
        print(f"\n🎯 AUTO-DETECTION RESULT:")
        print(f"  Selected Model:     {detailed_result['selected_model'].upper()}")
        print(f"  Predicted Label:    {detailed_result['prediction_label']}")
        print(f"  Confidence:         {detailed_result['prediction_confidence']:.2%}")
        print(f"  Model Score:        {detailed_result['selected_model_score']:.4f}")
        print(f"  Is Confident:       {'✅ YES' if detailed_result['auto_detect_confident'] else '⚠️  NO'}")
        
        if not detailed_result['auto_detect_confident'] and 'confidence_reasons' in detailed_result:
            print(f"\n⚠️  CONFIDENCE ISSUES:")
            for reason in detailed_result['confidence_reasons']:
                print(f"    - {reason}")
        
        print(f"\n📊 ALL MODEL SCORES (Combined Scoring):")
        print(f"  {'Model':<10} {'Score':<12} {'Bar Chart'}")
        print(f"  {'-'*10} {'-'*12} {'-'*40}")
        
        sorted_scores = sorted(
            detailed_result['all_model_scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        max_score = sorted_scores[0][1] if sorted_scores else 1.0
        
        for model_name, score in sorted_scores:
            bar_length = int((score / max_score) * 30)
            bar = '█' * bar_length
            marker = "⭐" if model_name == detailed_result['selected_model'] else "  "
            print(f"{marker} {model_name.capitalize():<10} {score:<12.4f} {bar}")
        
        # Run analyze_image_auto for comparison
        print(f"\n{'='*80}")
        print(f"FULL AUTO-ANALYSIS (with advice)")
        print(f"{'='*80}")
        
        analysis = analyze_image_auto(b64_image, require_threshold=threshold)
        
        print(f"\n🔬 ANALYSIS RESULT:")
        print(f"  Crop:          {analysis['selected_model'].capitalize()}")
        print(f"  Condition:     {analysis['label']}")
        print(f"  Confidence:    {analysis['confidence']:.2%}")
        print(f"  Description:   {analysis['description']}")
        print(f"  Advice:        {analysis['advice']}")
        print(f"  Is Confident:  {'✅ YES' if analysis['auto_detect_confident'] else '⚠️  NO'}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ ERROR in auto-detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python debug_vision.py <image_path> [--crop <crop_name>] [--threshold <value>]")
        print("\nExamples:")
        print("  python debug_vision.py test_images/cotton_healthy.jpg")
        print("  python debug_vision.py test_images/wheat_rust.jpg --crop wheat")
        print("  python debug_vision.py test_images/corn_blight.jpg --threshold 0.6")
        sys.exit(1)
    
    image_path = sys.argv[1]
    specific_crop = None
    threshold = 0.50
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--crop' and i + 1 < len(sys.argv):
            specific_crop = sys.argv[i + 1].lower()
            i += 2
        elif sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# VISION MODEL DEBUG TOOL")
    print(f"{'#'*80}")
    print(f"\n📁 Image Path: {image_path}")
    print(f"📏 Threshold:  {threshold}")
    if specific_crop:
        print(f"🌾 Testing:    {specific_crop.upper()} model only")
    else:
        print(f"🌾 Testing:    ALL models + auto-detection")
    
    # Convert image to base64
    try:
        b64_image = image_to_base64(image_path)
        print(f"✅ Image loaded successfully")
        
        # Get image info
        from PIL import Image
        with Image.open(image_path) as img:
            print(f"   Size: {img.size}, Mode: {img.mode}, Format: {img.format}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # Run tests
    if specific_crop:
        # Test single model
        if specific_crop not in ["cotton", "wheat", "corn", "rice"]:
            print(f"❌ Error: Invalid crop name. Must be one of: cotton, wheat, corn, rice")
            sys.exit(1)
        test_single_model(specific_crop, b64_image)
    else:
        # Test all models
        test_all_models(b64_image)
        
        # Test auto-detection
        test_auto_detection(b64_image, threshold=threshold)
    
    print(f"\n{'#'*80}")
    print(f"# DEBUG COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()