import sys
import os
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.transcriber import AudioTranscriber
from src.schemas import NoteEvent, StemType, MIDIData

def create_mock_notes(bpm=120):
    """Create notes for a C Major scale"""
    notes = []
    beat_dur = 60.0 / bpm
    
    # C Major Scale: C4, D4, E4, F4, G4, A4, B4, C5
    # MIDI: 60, 62, 64, 65, 67, 69, 71, 72
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    
    for i, p in enumerate(pitches):
        notes.append(NoteEvent(
            pitch=p,
            start_time=i * beat_dur,
            end_time=(i + 0.8) * beat_dur, # 0.8 duration to leave gap
            velocity=100
        ))
        
    return notes

def test_analyze_features():
    transcriber = AudioTranscriber()
    
    notes = create_mock_notes(bpm=120)
    midi_data = {
        StemType.VOCALS: MIDIData(
            stem_type=StemType.VOCALS,
            notes=notes
        )
    }
    
    print("Running feature analysis...")
    features = transcriber.analyze_features(midi_data)
    
    print("\nResults:")
    print(f"BPM: {features.bpm}")
    print(f"Key: {features.key}")
    print(f"Duration: {features.duration_seconds}")
    
    if features.stem_analyses:
        print("\nStem Analyses:")
        for k, v in features.stem_analyses.items():
            print(f"{k}: {v.description}")
            print(f"  Density: {v.note_density}")
            print(f"  Range: {v.pitch_range}")
            
    # Assertions
    # Note: Short sequence might make BPM detection unstable, but let's check basic execution
    
    # C Major profile should match well
    # Since we only provided diatonic notes of C Major, C Major should be the top candidate
    print(f"Detected Key: {features.key}")
    
    assert features.duration_seconds > 0
    assert "vocals" in features.stem_analyses

if __name__ == "__main__":
    test_analyze_features()
