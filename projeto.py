import json
import os
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="QAAI - Quality Assurance AI",
    page_icon="🧪",
    layout="wide"
)

# Adicionando CSS para personalizar o botão para cor branca
st.markdown("""
    <style>
    .stButton>button {
        color: white; 
        border: 1px solid white; 
        padding: 10px 24px;
    }
    .stButton>button:hover {
        color: black;
        background: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("QAAI - Quality Assurance AI 🧪")
st.caption("Geração Inteligente de Casos de Teste")

# Definindo a estrutura dos casos de teste
class TestCase(BaseModel):
    title: str = Field(description="Título do caso de teste")
    description: str = Field(description="Descrição detalhada do caso de teste")
    preconditions: List[str] = Field(description="Lista de pré-condições necessárias")
    steps: List[str] = Field(description="Lista de passos do teste")
    expected_results: List[str] = Field(description="Lista de resultados esperados")
    test_type: str = Field(description="Tipo do teste (unitário, integração, funcional, etc)")

# Configuração do modelo
def setup_llm():
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
    )

# Template do prompt
test_generation_template = """
Você é um especialista em QA e automação de testes. Com base na descrição da funcionalidade fornecida, 
gere casos de teste detalhados e estruturados.

Descrição da funcionalidade:
{functionality_description}

Tipo de teste desejado: {test_type}

Gere um caso de teste detalhado seguindo estas diretrizes:
- Seja específico e claro
- Inclua pré-condições necessárias
- Forneça passos detalhados
- Especifique os resultados esperados
- Considere cenários positivos e negativos

{format_instructions}
"""

# Configuração do parser
parser = PydanticOutputParser(pydantic_object=TestCase)

# Inicialização do histórico
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []

# Interface do usuário
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Descrição da Funcionalidade")
        functionality = st.text_area(
            "Descreva a funcionalidade para a qual você precisa gerar casos de teste:",
            height=150
        )
        
        test_type = st.selectbox(
            "Tipo de Teste",
            ["Funcional", "Unitário", "Integração", "E2E", "Aceitação"]
        )
        
        if st.button("Gerar Casos de Teste"):
            if functionality:
                try:
                    llm = setup_llm()
                    prompt = ChatPromptTemplate.from_template(template=test_generation_template)
                    
                    messages = prompt.format_messages(
                        functionality_description=functionality,
                        test_type=test_type,
                        format_instructions=parser.get_format_instructions()
                    )
                    
                    output = llm.invoke(messages)
                    test_case = parser.parse(output.content)
                    st.session_state.test_cases.append(test_case)
                    
                    st.success("Caso de teste gerado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao gerar caso de teste: {str(e)}")
            else:
                st.warning("Por favor, descreva a funcionalidade primeiro.")

    with col2:
        st.subheader("Casos de Teste Gerados")
        if st.session_state.test_cases:
            # Botão para exportar todos os casos
            if st.button("Exportar Todos os Casos", key="export_all"):
                all_tests = [test.model_dump() for test in st.session_state.test_cases]
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Todos (JSON)",
                    data=json.dumps(all_tests, indent=2, ensure_ascii=False),
                    file_name=f"all_test_cases_{current_date}.json",
                    mime="application/json",
                    key="download_all"
                )
            
            # Lista individual de casos de teste
            for i, test in enumerate(st.session_state.test_cases):
                with st.expander(f"Caso de Teste {i+1}: {test.title}"):
                    st.write("**Descrição:**")
                    st.write(test.description)
                    
                    st.write("**Pré-condições:**")
                    for pre in test.preconditions:
                        st.write(f"- {pre}")
                    
                    st.write("**Passos:**")
                    for step in test.steps:
                        st.write(f"- {step}")
                    
                    st.write("**Resultados Esperados:**")
                    for result in test.expected_results:
                        st.write(f"- {result}")
                    
                    st.write(f"**Tipo de Teste:** {test.test_type}")
                    
                    # Botão para exportar o caso de teste
                    if st.button(f"Exportar Caso {i+1}", key=f"export_{i}"):
                        test_dict = test.model_dump()
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(test_dict, indent=2, ensure_ascii=False),
                            file_name=f"test_case_{i+1}.json",
                            mime="application/json",
                            key=f"download_{i}"
                        )