#!/usr/bin/env python3
"""
Quick test script for single image processing.
Tests the full pipeline on a static image.
"""

import cv2
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detector import ObjectDetector
from tracker import Tracker
from visualization import draw_tracks, draw_segmentation_masks


def test_image(image_path: str, output_path: str = None, show: bool = True):
    """
    Test detection and tracking on a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output (optional)
        show: Whether to display result
    """
    print("\n" + "="*60)
    print("  IMAGE DETECTION & TRACKING TEST")
    print("="*60)
    print(f"\nInput: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    h, w = img.shape[:2]
    print(f"✓ Loaded image: {w}x{h}")
    
    # Initialize detector
    print("\n🔍 Initializing detector...")
    detector = ObjectDetector(
        model_name='yolov8s.pt',
        conf_threshold=0.25,
        use_segmentation=True
    )
    
    # Detect
    print("🔍 Running detection...")
    detections = detector.detect(img)
    print(f"✓ Found {len(detections)} objects:")
    for det in detections:
        print(f"  • {det.class_label} (confidence: {det.confidence:.2f})")
    
    # Initialize tracker
    print("\n🎯 Initializing tracker...")
    tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
    
    # Track (simulate a few frames for stable IDs)
    print("🎯 Running tracking...")
    for i in range(5):
        tracks = tracker.update(detections)
    
    print(f"✓ Created {len(tracks)} tracks:")
    for track in tracks:
        print(f"  • Track {track.id}: {track.class_label}")
    
    # Visualize
    print("\n🎨 Creating visualization...")
    processed = img.copy()
    
    # Add segmentation masks
    if any(d.mask is not None for d in detections):
        draw_segmentation_masks(processed, detections, alpha=0.4)
        print("✓ Added segmentation masks")
    
    # Add tracking visualization
    draw_tracks(processed, tracks, show_speed=True, show_trajectory=True)
    print("✓ Added tracking visualization")
    
    # Save
    if output_path is None:
        output_path = f"test_output_{Path(image_path).stem}.jpg"
    
    cv2.imwrite(output_path, processed)
    print(f"\n✅ Saved result: {output_path}")
    
    # Display
    if show:
        print("\n👁️  Displaying result...")
        print("   Press any key to close")
        
        # Resize if too large
        display_img = processed
        if w > 1200:
            scale = 1200 / w
            new_w = 1200
            new_h = int(h * scale)
            display_img = cv2.resize(processed, (new_w, new_h))
        
        cv2.imshow('Detection & Tracking Result', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print(f"  Detections: {len(detections)}")
    print(f"  Tracks: {len(tracks)}")
    print(f"  Output: {output_path}")
    print("="*60 + "\n")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test detection and tracking on a single image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('image', nargs='?',
                        default='demos/original/DSCF0880.JPG',
                        help='Input image path')
    parser.add_argument('--output', '-o', default=None,
                        help='Output image path')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display result')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"\n❌ Image not found: {args.image}")
        print("\nAvailable images:")
        demo_dir = Path('demos/original')
        if demo_dir.exists():
            for img in demo_dir.glob('*.JPG') + demo_dir.glob('*.jpg'):
                print(f"  • {img}")
        else:
            print("  None found in demos/original/")
        return 1
    
    # Test image
    success = test_image(args.image, args.output, not args.no_display)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
