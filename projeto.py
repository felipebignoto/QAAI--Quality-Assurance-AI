import json
import os
import re
from datetime import datetime
from typing import List, Dict, Union, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="QAAI - Quality Assurance AI",
    page_icon="🧪",
    layout="wide"
)

# Carregando o CSS externo
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # CSS básico para garantir alguma formatação
    st.markdown("""
    <style>
    .generate-btn button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .custom-btn-container button {
        margin-right: 10px;
    }
    .test-container {
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 10px;
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
    test_code: str = Field(description="Código de implementação do teste")

# Classe para resposta combinada de validação e geração com múltiplos casos
class ValidationWithTestCases(BaseModel):
    is_valid: bool = Field(description="Indica se a descrição é válida para gerar casos de teste")
    message: str = Field(description="Mensagem de erro ou aviso se a descrição não for válida")
    test_cases: List[TestCase] = Field(description="Lista de casos de teste gerados se a descrição for válida", default=[])

# Configuração do modelo
def setup_llm():
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
    )

# Função para formatar o código corretamente
def format_code(code_string):
    """
    Formata o código de teste para exibição adequada,
    substituindo os caracteres de escape por quebras de linha reais.
    """
    # Se o código já estiver formatado corretamente, retorna como está
    if "\n" in code_string and not "\\n" in code_string:
        return code_string
    
    # Substitui \\n por \n para garantir que todos os caracteres de escape sejam processados
    code_string = code_string.replace("\\n", "\n")
    
    # Remove possíveis aspas extras no início e no final
    code_string = code_string.strip()
    if code_string.startswith('"') and code_string.endswith('"'):
        code_string = code_string[1:-1]
    if code_string.startswith("'") and code_string.endswith("'"):
        code_string = code_string[1:-1]
    
    return code_string

# Template do prompt para geração de múltiplos casos de teste com instruções explícitas de formatação
multi_test_template = """
Você é um especialista em QA e automação de testes. Com base na descrição da funcionalidade fornecida,
primeiro avalie se a descrição contém informações suficientes para gerar casos de teste.
Use um critério menos rigoroso - se a descrição fizer o mínimo de sentido para entender a funcionalidade, considere-a válida.

Descrição da funcionalidade:
{functionality_description}

Tipo de teste desejado: {test_type}

Linguagem de programação preferida: {programming_language}

ETAPA 1: Analise brevemente se a descrição faz o mínimo de sentido para gerar casos de teste.
- Se a descrição contiver pelo menos o básico sobre a funcionalidade, considere-a válida
- Só rejeite descrições totalmente inadequadas ou vazias de conteúdo

ETAPA 2: Se a descrição for válida, GERE MÚLTIPLOS CASOS DE TESTE que sejam relevantes para cobrir diferentes aspectos da funcionalidade.
Você deve gerar pelo menos 2 casos de teste diferentes quando a funcionalidade for complexa o suficiente para exigir vários cenários de teste.

Para cada caso de teste, siga estas diretrizes:
- Seja específico e claro
- Inclua pré-condições necessárias
- Forneça passos detalhados
- Especifique os resultados esperados
- Considere cenários positivos e negativos
- Gere um código de implementação do teste na linguagem de programação especificada ({programming_language}), utilizando a tecnologia mais adequada para o tipo de teste:
  * Para testes unitários: utilize frameworks como pytest, JUnit, Jest, etc.
  * Para testes de integração: utilize ferramentas como RestAssured, Supertest, etc.
  * Para testes funcionais/E2E: utilize Selenium, Cypress, Playwright, etc.
  * Adapte os frameworks de acordo com a linguagem escolhida

IMPORTANTE SOBRE O CÓDIGO DE TESTE:
- Escreva o código na linguagem {programming_language}
- Inclua quebras de linha reais no código, não use caracteres de escape como \\n
- Formate o código adequadamente com indentação correta
- Não coloque o código entre aspas ou escape characters
- O código deve estar pronto para ser executado

Aborde diferentes cenários como:
- Caminho feliz (cenário principal)
- Tratamento de erros e exceções
- Casos limite (boundary values)
- Casos de validação
- Casos de segurança (quando relevante)

IMPORTANTE: Sua resposta deve ser um JSON válido que segue exatamente o formato abaixo. Não inclua explicações adicionais, texto ou markdown fora do JSON.
Cada campo precisa estar devidamente formatado para a correta deserialização. Não adicione campos extras além dos especificados no formato.

{format_instructions}

Se a descrição não for válida, retorne apenas um objeto JSON com is_valid=false, uma mensagem explicativa em message, e test_cases como um array vazio.
"""

# Configuração do parser para múltiplos casos
parser = PydanticOutputParser(pydantic_object=ValidationWithTestCases)

# Parser personalizado para lidar com falhas de parsing do JSON
def parse_llm_response(content: str) -> ValidationWithTestCases:
    """
    Tenta analisar a resposta do LLM e extrair um JSON válido, mesmo se estiver mal formatado.
    Implementa várias estratégias de correção e recuperação de erros.
    """
    # Primeira tentativa: usar o parser padrão
    try:
        return parser.parse(content)
    except Exception as e:
        st.write(f"Erro no parsing inicial: {str(e)}")
        
        # Segunda tentativa: extrair o JSON se estiver dentro de blocos de código markdown
        try:
            # Padrão para extrair JSON de blocos de código markdown
            json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            match = re.search(json_pattern, content)
            if match:
                json_str = match.group(1).strip()
                parsed_json = json.loads(json_str)
                return ValidationWithTestCases.model_validate(parsed_json)
        except Exception as e:
            st.write(f"Erro ao extrair JSON de blocos de código: {str(e)}")
        
        # Terceira tentativa: procurar por um objeto JSON válido em qualquer lugar do texto
        try:
            # Procurar por qualquer coisa que pareça um objeto JSON 
            potential_json_pattern = r"\{[\s\S]*\}"
            match = re.search(potential_json_pattern, content)
            if match:
                json_str = match.group(0)
                parsed_json = json.loads(json_str)
                return ValidationWithTestCases.model_validate(parsed_json)
        except Exception as e:
            st.write(f"Erro ao extrair potencial JSON do texto: {str(e)}")
        
        # Última tentativa: criar uma resposta de erro com a mensagem original
        return ValidationWithTestCases(
            is_valid=False,
            message=f"Não foi possível analisar a resposta. Erro de formato: {content[:200]}...",
            test_cases=[]
        )

# Inicialização do histórico
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []

# Função para processar entrada com mecanismo de retry
def process_input(functionality_description, test_type, programming_language, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            llm = setup_llm()
            prompt = ChatPromptTemplate.from_template(template=multi_test_template)
            
            messages = prompt.format_messages(
                functionality_description=functionality_description,
                test_type=test_type,
                programming_language=programming_language,
                format_instructions=parser.get_format_instructions()
            )
            
            output = llm.invoke(messages)
            
            # Usar o parser personalizado para lidar com potenciais problemas de formato
            result = parse_llm_response(output.content)
            
            # Formatar o código de teste em cada caso gerado
            if result.is_valid and result.test_cases:
                for test_case in result.test_cases:
                    test_case.test_code = format_code(test_case.test_code)
            
            # Se o parsing for bem-sucedido e tivermos casos de teste ou uma mensagem de erro válida, retorne
            if result.is_valid and result.test_cases:
                return result
            elif not result.is_valid and result.message:
                return result
            
            # Se chegamos aqui, o parsing foi bem-sucedido, mas não temos casos de teste válidos
            # Se não for a última tentativa, tentaremos novamente
            if attempt < max_retries:
                continue
            else:
                return ValidationWithTestCases(
                    is_valid=False,
                    message="Após várias tentativas, não foi possível gerar casos de teste válidos. Por favor, forneça uma descrição mais detalhada.",
                    test_cases=[]
                )
                
        except Exception as e:
            # Se não for a última tentativa, tentaremos novamente
            if attempt < max_retries:
                continue
            else:
                return ValidationWithTestCases(
                    is_valid=False,
                    message=f"Erro ao processar entrada após {max_retries+1} tentativas: {str(e)}",
                    test_cases=[]
                )

# Interface do usuário
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Descrição ou Código da Funcionalidade")
        functionality = st.text_area(
            "Descreva a funcionalidade ou cole o código-fonte para o qual deseja gerar casos de teste:",
            height=150,
            help="Você pode inserir uma descrição textual ou o código-fonte da função/componente que deseja testar."
        )
        
        # Organizando os controles em colunas
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            test_type = st.selectbox(
                "Abordagem de Teste",
                ["Funcional", "Unitário", "Integração", "E2E", "Aceitação"],
                help="Selecione a abordagem de teste que melhor se aplica ao seu contexto."
            )
        
        with input_col2:
            # Lista de linguagens de programação comuns para testes
            programming_language = st.selectbox(
                "Linguagem de Programação",
                ["Python", "Java", "JavaScript", "TypeScript", "C#", "Ruby", "Go", "PHP", "Outra"],
                index=0,  # Python como padrão
                help="Selecione a linguagem de programação para o código de teste. Este campo é obrigatório."
            )
            
            # Opção para personalizar a linguagem se "Outra" for selecionada
            if programming_language == "Outra":
                custom_language = st.text_input("Especifique a linguagem:")
                if custom_language:
                    programming_language = custom_language
        
        with st.container():
            # Aplicando a classe CSS para o botão de gerar casos de teste
            generate_button_col = st.container()
            with generate_button_col:
                st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
                generate_pressed = st.button("Gerar Casos de Teste")
                st.markdown('</div>', unsafe_allow_html=True)
                
            if generate_pressed:
                if not functionality:
                    st.warning("Por favor, forneça uma descrição ou código da funcionalidade.")
                elif programming_language == "Outra" and not custom_language:
                    st.warning("Por favor, especifique a linguagem de programação.")
                else:
                    # Exibir mensagem de processamento
                    with st.spinner("Processando entrada..."):
                        result = process_input(functionality, test_type, programming_language)
                    
                    if result.is_valid and result.test_cases:
                        # Adiciona todos os casos de teste gerados à sessão
                        for test_case in result.test_cases:
                            st.session_state.test_cases.append(test_case)
                        
                        st.success(f"{len(result.test_cases)} caso(s) de teste gerado(s) com sucesso!")
                    else:
                        st.error("Não foi possível gerar os casos de teste:")
                        st.warning(result.message)

    with col2:
        st.markdown('<div class="test-container">', unsafe_allow_html=True)
        st.subheader("Casos de Teste Gerados")
        if st.session_state.test_cases:
            # Botão para exportar todos os casos
            st.markdown('<div class="custom-btn-container">', unsafe_allow_html=True)
            export_all = st.button("Exportar Todos os Casos", key="export_all")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if export_all:
                # Assegura que o código está formatado corretamente antes de exportar
                formatted_tests = []
                for test in st.session_state.test_cases:
                    formatted_test = test.model_dump()
                    formatted_test["test_code"] = format_code(test.test_code)
                    formatted_tests.append(formatted_test)
                
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(formatted_tests, indent=2, ensure_ascii=False),
                    file_name=f"all_test_cases_{current_date}.json",
                    mime="application/json",
                    key="download_all_json"
                )
                
                # Opção para baixar todos os códigos em um arquivo ZIP
                try:
                    import io
                    import zipfile
                    
                    # Criar arquivo ZIP na memória
                    zip_io = io.BytesIO()
                    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                        for i, test in enumerate(st.session_state.test_cases):
                            # Determinar extensão apropriada para a linguagem
                            extension = ".py"  # Padrão para Python
                            code = format_code(test.test_code)
                            
                            # Detectar linguagem a partir do conteúdo do código
                            if "public class" in code or "System.out.println" in code:
                                extension = ".java"
                            elif "function" in code and ("=>" in code or "document." in code):
                                extension = ".js"
                            elif "namespace" in code or "public void" in code:
                                extension = ".cs"
                            
                            # Adicionar arquivo ao ZIP
                            zip_file.writestr(f"test_code_{i+1}{extension}", code)
                    
                    # Retornar ao início do BytesIO para leitura
                    zip_io.seek(0)
                    
                    # Botão para baixar o ZIP
                    st.download_button(
                        label="Download Códigos",
                        data=zip_io,
                        file_name=f"all_test_codes_{current_date}.zip",
                        mime="application/zip",
                        key="download_all_code"
                    )
                except Exception as e:
                    st.error(f"Erro ao criar arquivo ZIP: {str(e)}")
            
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
                    
                    st.write("**Código de Implementação:**")
                    # Garantir que o código está formatado corretamente para exibição
                    formatted_code = format_code(test.test_code)
                    
                    # Detecção de linguagem baseada no conteúdo do código
                    language = "python"  # Padrão
                    if "public class" in formatted_code or "System.out.println" in formatted_code:
                        language = "java"
                    elif "function" in formatted_code and ("=>" in formatted_code or "document." in formatted_code):
                        language = "javascript"
                    elif "namespace" in formatted_code or "public void" in formatted_code:
                        language = "csharp"
                    
                    st.code(formatted_code, language=language)
                    
                    # Botão para exportar o caso de teste
                    st.markdown(f'<div class="custom-btn-container">', unsafe_allow_html=True)
                    export_case = st.button(f"Exportar Caso {i+1}", key=f"export_{i}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if export_case:
                        test_dict = test.model_dump()
                        # Garantir que o código está formatado corretamente para exportação
                        test_dict["test_code"] = format_code(test.test_code)
                        
                        # Opções de download para caso individual
                        cols = st.columns(2)
                        with cols[0]:
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(test_dict, indent=2, ensure_ascii=False),
                                file_name=f"test_case_{i+1}.json",
                                mime="application/json",
                                key=f"download_json_{i}"
                            )
                        
                        # Extensão de arquivo baseada na linguagem
                        extension = ".py"  # Padrão para Python
                        if language == "java":
                            extension = ".java"
                        elif language == "javascript":
                            extension = ".js"
                        elif language == "csharp":
                            extension = ".cs"
                        
                        with cols[1]:
                            # Adicionar opção para baixar apenas o código de teste
                            st.download_button(
                                label="Download Código",
                                data=format_code(test.test_code),
                                file_name=f"test_code_{i+1}{extension}",
                                mime="text/plain",
                                key=f"download_code_{i}"
                            )
        st.markdown('</div>', unsafe_allow_html=True)