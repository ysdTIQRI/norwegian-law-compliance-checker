import streamlit as st
import tempfile
from pathlib import Path

from src.rag_pipeline import run_analysis


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Norwegian Law Checker",
    page_icon="⚖️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */

    .main-title {
        color: #0A66C2;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Section headings */

    .section-title {
        color: #0A66C2;
        font-size: 21px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    /* Buttons */

    div.stButton > button {
        background-color: #0A66C2;
        color: white;
        border-radius: 7px;
        border: none;
        height: 45px;
        font-size: 15px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #084f96;
        color: white;
        border: none;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        border-radius: 8px;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">⚖️ Norwegian Law Checker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Employment Contract Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">1. Upload Employment Document</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an employment document",
    type=["pdf", "csv", "xlsx", "xls"],
    help="Supported formats: PDF, CSV, XLSX and XLS."
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Uploaded: {uploaded_file.name}")

    with col2:
        st.caption(
            f"File size: {uploaded_file.size / 1024:.2f} KB"
        )


st.divider()


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">2. Ask a Question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Enter your question or request a compliance check:",
    value=(
        "Does this employment contract comply with Norwegian "
        "employment law? Identify potential issues and explain "
        "the findings using relevant legal sections and evidence."
    ),
    height=110
)


# --------------------------------------------------
# ANALYSE BUTTON
# --------------------------------------------------

st.divider()

analyse_button = st.button(
    "Analyse Employment Document",
    use_container_width=True
)


# --------------------------------------------------
# RUN RAG PIPELINE
# --------------------------------------------------

if analyse_button:

    progress = st.empty()

    if uploaded_file is None:

        st.warning("Please upload an employment document.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        temp_document_path = None

        try:
            # Keep the original file extension so the pipeline
            # can identify whether the file is PDF, CSV or Excel.
            file_suffix = Path(uploaded_file.name).suffix.lower()

            with tempfile.NamedTemporaryFile(
                suffix=file_suffix,
                delete=False
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_document_path = Path(temp_file.name)

            with st.spinner(
                "Analysing your document. Please wait..."
            ):

                progress.info(
                    "Extracting document content and performing "
                    "security validation and legal analysis..."
                )

                result = run_analysis(
                    temp_document_path,
                    question
                )

            progress.empty()

            # --------------------------------------------------
            # SECURITY VALIDATION
            # --------------------------------------------------

            if not result["allowed"]:

                st.error(
                    "Document processing was blocked by security validation."
                )

            else:

                st.success("Document analysis completed successfully.")

                st.divider()

                st.markdown(
                    '<div class="section-title">3. Security Validation</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)

                # page_count is retained by the current pipeline for
                # compatibility. For CSV/Excel it represents extracted
                # document sections or rows, not PDF pages.
                document_count = result.get(
                    "document_location_count",
                    result.get("page_count", 0)
                )

                col1.metric(
                    "Document Pages / Rows",
                    document_count
                )

                col2.metric(
                    "PII Detected",
                    "Yes" if result["pii_detected"] else "No"
                )

                col3.metric(
                    "PII Masking",
                    "Completed" if result["masking_completed"]
                    else "Not required"
                )

                st.success("OPA security validation: Allowed")

                st.divider()

                # --------------------------------------------------
                # AI ANALYSIS
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">4. Compliance Analysis</div>',
                    unsafe_allow_html=True
                )

                st.warning(
                    "AI-generated findings are preliminary. "
                    "They are not confirmed legal conclusions or legal advice."
                )

                st.markdown(result["analysis"])

                st.divider()

                # --------------------------------------------------
                # CONTRACT / DOCUMENT EVIDENCE
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">5. Document Evidence</div>',
                    unsafe_allow_html=True
                )

                for index, item in enumerate(
                    result["contract_results"],
                    start=1
                ):

                    chunk = item["chunk"]
                    location = chunk.get("page", "Location not available")

                    with st.expander(
                        f"Document Evidence {index} - {location}"
                    ):

                        st.write(chunk["text"])

                        st.caption(
                            f"Similarity score: {item['score']:.3f}"
                        )

                st.divider()

                # --------------------------------------------------
                # LEGAL REFERENCES
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">6. Retrieved Legal References</div>',
                    unsafe_allow_html=True
                )

                for index, item in enumerate(
                    result["legal_results"],
                    start=1
                ):

                    with st.expander(
                        f"Legal Reference {index} - {item['reference']}"
                    ):

                        st.write(item["text"])

                        st.caption(
                            f"Similarity score: {item['score']:.3f}"
                        )

                st.divider()

                st.success("Document analysis completed successfully.")

        except Exception as error:

            progress.empty()

            st.error(
                "An error occurred during document analysis."
            )

            st.exception(error)
        finally:
            # Always attempt to remove the temporary uploaded file.
            if temp_document_path is not None:
                try:
                    temp_document_path.unlink(missing_ok=True)

                except PermissionError as error:
                    st.warning(
                        f"Could not delete temporary file: {error}"
                    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Norwegian Law Checker | AI-powered Legal Document Analysis "
    "| Prototype - Not Legal Advice"
)
