"""
RCT-Reviewer v1.0.0 - Main Streamlit Application

Modernized clinical trial analysis system for automatic extraction
of PICO data and Risk of Bias assessment.

Run with: streamlit run rct_reviewer/app.py

Copyright (C) 2024-2026 Vihaan Sahu
Based on RobotReviewer by Iain Marshall, Joel Kuiper, Byron Wallace
"""

import logging
import sys
import time
from typing import Any

import streamlit as st

from rct_reviewer import __version__
from rct_reviewer.config import settings
from rct_reviewer.core.models import DocumentAnalysis
from rct_reviewer.core.pdf_parser import PDFParser
from rct_reviewer.ml.rct_classifier import RCTClassifier
from rct_reviewer.ml.pico_extractor import PICOExtractor
from rct_reviewer.ml.bias_assessor import BiasAssessor


logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def setup_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=settings.ui.page_title,
        page_icon=settings.ui.page_icon,
        layout=settings.ui.layout,
        initial_sidebar_state="expanded",
    )
    

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header() -> None:
    """Display application header."""
    st.markdown('<div class="main-header">🔬 RCT-Reviewer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Automatic extraction of PICO data and Risk of Bias assessment '
        'from clinical trial reports</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Version {__version__} | Python 3.13+ | Transformer-based analysis")
    st.divider()


def display_sidebar() -> None:
    """Display sidebar with settings and information."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        

        st.markdown("### Analysis Options")
        
        run_rct = st.checkbox(
            "RCT Classification",
            value=True,
            help="Classify whether documents describe Randomized Controlled Trials",
        )
        
        run_pico = st.checkbox(
            "PICO Extraction",
            value=True,
            help="Extract Population, Intervention, and Outcome information",
        )
        
        run_bias = st.checkbox(
            "Risk of Bias",
            value=True,
            help="Assess Risk of Bias using Cochrane RoB 1.0 domains",
        )
        
        st.divider()
        

        st.markdown("### Model Settings")
        
        threshold_type = st.selectbox(
            "RCT Threshold",
            options=["balanced", "precise", "sensitive"],
            index=1,
            help="balanced: balanced precision/recall\nprecise: high precision\nsensitive: high recall",
        )
        
        pico_top_k = st.slider(
            "PICO sentences per domain",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top sentences to extract for each PICO domain",
        )
        
        bias_top_k = st.slider(
            "Bias evidence sentences",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of evidence sentences per bias domain",
        )
        
        st.divider()
        
   
        st.markdown("### PDF Parsing")
        
        try:
            from rct_reviewer.core.pdf_parser import GrobidClient
            grobid = GrobidClient()
            grobid_available = grobid.is_available()
        except Exception:
            grobid_available = False
        
        if grobid_available:
            st.success("✅ GROBID connected")
        else:
            st.warning("⚠️ GROBID not available\nUsing PyMuPDF fallback")
            st.caption("For better results, start GROBID:\n`docker run -d -p 8070:8070 lfoppiano/grobid:latest`")
        
        st.divider()
        

        if settings.ui.show_citation:
            st.markdown("### 📝 Citation")
            st.caption(
                "Risk of Bias automation by RCT-Reviewer "
                "(based on RobotReviewer by Marshall et al., 2017)"
            )
        
        return {
            "run_rct": run_rct,
            "run_pico": run_pico,
            "run_bias": run_bias,
            "threshold_type": threshold_type,
            "pico_top_k": pico_top_k,
            "bias_top_k": bias_top_k,
        }


def initialize_models(options: dict[str, Any]) -> dict[str, Any]:
    """Initialize ML models based on options."""
    models: dict[str, Any] = {}
    
    if options.get("run_rct"):
        with st.spinner("Loading RCT classifier..."):
            try:
                models["rct"] = RCTClassifier()
                models["rct"].load()
                log.info("RCT classifier loaded")
            except Exception as e:
                st.error(f"Failed to load RCT classifier: {e}")
                log.error(f"RCT classifier load failed: {e}")
    
    if options.get("run_pico"):
        with st.spinner("Loading PICO extractor..."):
            try:
                models["pico"] = PICOExtractor(top_k=options.get("pico_top_k", 3))
                models["pico"].load()
                log.info("PICO extractor loaded")
            except Exception as e:
                st.error(f"Failed to load PICO extractor: {e}")
                log.error(f"PICO extractor load failed: {e}")
    
    if options.get("run_bias"):
        with st.spinner("Loading Bias assessor..."):
            try:
                models["bias"] = BiasAssessor(top_k=options.get("bias_top_k", 3))
                models["bias"].load()
                log.info("Bias assessor loaded")
            except Exception as e:
                st.error(f"Failed to load Bias assessor: {e}")
                log.error(f"Bias assessor load failed: {e}")
    
    return models


def analyze_documents(
    pdf_files: list[tuple[bytes, str]],
    models: dict[str, Any],
    options: dict[str, Any],
) -> list[DocumentAnalysis]:
    """Analyze PDF documents."""
    results: list[DocumentAnalysis] = []
    total = len(pdf_files)
    

    parser = PDFParser()
    

    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="Starting analysis...")
    
    with status_container:
        status_text = st.empty()
    
    for i, (pdf_bytes, filename) in enumerate(pdf_files):
        start_time = time.time()
        
  
        progress = (i / total) * 100
        progress_bar.progress(
            progress / 100,
            text=f"Processing {i + 1}/{total}: {filename}",
        )
        status_text.markdown(f"📄 **{filename}**")
        
        try:
  
            status_text.markdown(f"📄 **{filename}** — Parsing PDF...")
            doc = parser.parse(pdf_bytes, filename)
            
            if doc.parse_error:
                doc.warnings.append("PDF parsing failed")
                results.append(doc)
                continue
            
  
            if "rct" in models and options.get("run_rct"):
                status_text.markdown(f"📄 **{filename}** — Classifying RCT...")
                title = doc.publication.title or ""
                abstract = doc.publication.abstract or ""
                
                doc.rct = models["rct"].predict(
                    title=title,
                    abstract=abstract,
                    threshold_type=options.get("threshold_type", "balanced"),
                )
                doc.models_used.append(f"rct:{models['rct'].model_name}")
            
         
            if "pico" in models and options.get("run_pico"):
                status_text.markdown(f"📄 **{filename}** — Extracting PICO...")
                
          
                doc.pico_spans = models["pico"].extract_spans(
                    title=doc.publication.title,
                    abstract=doc.publication.abstract,
                )
                
            
                if doc.full_text:
                    doc.pico = models["pico"].extract(
                        title=doc.publication.title,
                        abstract=doc.publication.abstract,
                        full_text=doc.full_text,
                    )
                
                doc.models_used.append(f"pico:{models['pico'].model_name}")
            

            if "bias" in models and options.get("run_bias"):
                status_text.markdown(f"📄 **{filename}** — Assessing Risk of Bias...")
                
                if doc.full_text:
                    doc.bias = models["bias"].assess(full_text=doc.full_text)
                    doc.models_used.append(f"bias:{models['bias'].model_name}")
                else:
                    doc.warnings.append("No full text available for Bias assessment")
            

            doc.processing_time_seconds = time.time() - start_time
            
            results.append(doc)
            log.info(f"Processed {filename} in {doc.processing_time_seconds:.2f}s")
            
        except Exception as e:
            log.error(f"Error processing {filename}: {e}", exc_info=True)
            error_doc = DocumentAnalysis(
                filename=filename,
                parse_error=True,
                warnings=[f"Processing error: {str(e)}"],
                processing_time_seconds=time.time() - start_time,
            )
            results.append(error_doc)
    

    progress_bar.progress(1.0, text="Analysis complete!")
    status_text.empty()
    
    return results


def main() -> None:
    """Main application entry point."""
    setup_page()
    display_header()
    

    options = display_sidebar()
    

    if "results" not in st.session_state:
        st.session_state.results: list[DocumentAnalysis] = []
    
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
        st.session_state.models: dict[str, Any] = {}
    

    from rct_reviewer.ui.components.pdf_uploader import pdf_uploader
    
    pdf_files = pdf_uploader()
    

    if pdf_files:
        analyze_clicked = st.button(
            "🚀 Analyze Documents",
            type="primary",
            use_container_width=True,
        )
        
        if analyze_clicked or st.session_state.get("analyzing"):
            st.session_state.analyzing = True
            
     
            if not st.session_state.models_loaded:
                st.session_state.models = initialize_models(options)
                st.session_state.models_loaded = True
            

            if st.session_state.models:
                st.session_state.results = analyze_documents(
                    pdf_files,
                    st.session_state.models,
                    options,
                )
                st.session_state.analyzing = False
                st.rerun()
    

    if st.session_state.results:
        st.divider()
        
        from rct_reviewer.ui.components.results_viewer import display_results
        from rct_reviewer.ui.components.export_buttons import export_results
        
        display_results(st.session_state.results)
        
        st.divider()
        export_results(st.session_state.results)
        

        if settings.ui.show_citation:
            from rct_reviewer.ui.components.citation_notice import show_citation_notice
            show_citation_notice()
    

    st.divider()
    st.caption(
        "RCT-Reviewer v1.0.0 | © 2024-2026 Vihaan Sahu | "
        "Based on RobotReviewer by Marshall, Kuiper, Wallace (2017) | "
        "GPL-3.0 License"
    )


if __name__ == "__main__":
    main()