import React, { useRef, useState } from "react";

export default function ImageUpload({ onUpload, onReset, analysing }) {
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    setFile(selected);
    onReset();

    const reader = new FileReader();
    reader.onload = (ev) => setPreview(ev.target.result);
    reader.readAsDataURL(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type.startsWith("image/")) {
      setFile(dropped);
      onReset();
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target.result);
      reader.readAsDataURL(dropped);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    onReset();
  };

  const handleSubmit = () => {
    if (file) onUpload(file);
  };

  return (
    <div className="card upload-wrapper">
      <h2>Upload a spiral or wave drawing</h2>

      <div
        className={`drop-zone ${preview ? "has-preview" : ""}`}
        onClick={() => !preview && fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {preview ? (
          <img src={preview} alt="Preview" className="upload-preview" />
        ) : (
          <div className="drop-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p>Drop an image here or click to browse</p>
            <span className="drop-hint">PNG, JPG, BMP supported</span>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/bmp"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      <div className="btn-row">
        <button className="btn btn-secondary" onClick={handleClear}>
          Clear
        </button>
        <button
          className="btn btn-primary"
          disabled={!file || analysing}
          onClick={handleSubmit}
        >
          {analysing ? (
            <>
              <span className="spinner" />
              Analysing…
            </>
          ) : (
            "Analyse Image"
          )}
        </button>
      </div>
    </div>
  );
}
