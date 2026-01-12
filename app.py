from flask import Flask, request, jsonify, Response
import requests
import json

app = Flask(__name__)

BACKEND_BASE_URL = "https://inference.do-ai.run/v1"
ORIGINAL_MODEL_ID = "anthropic-claude-opus-4.5"
REWROTE_MODEL_ID = "do-opus-4.5"

def transform_tool_use_to_text(content):
    """Transform tool_use content to natural text format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'tool_use':
                    # Skip tool_use items - they're not needed in text format
                    # The tool results will be in subsequent messages
                    continue
                elif item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                else:
                    text_parts.append(json.dumps(item))
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts) if text_parts else ''
    return str(content)

def transform_messages(messages):
    """Transform messages to ensure content is always a string."""
    transformed = []
    for msg in messages:
        new_msg = msg.copy()
        if 'content' in new_msg:
            new_msg['content'] = transform_tool_use_to_text(new_msg['content'])
        transformed.append(new_msg)
    return transformed

def transform_tool_choice(tool_choice):
    """Transform tool_choice from object format to string format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get('type')
        return choice_type if choice_type in ['none', 'auto', 'required'] else "auto"
    return "auto"

def transform_tools(tools):
    """Transform tools to OpenAI format."""
    if not tools:
        return tools
    transformed = []
    for tool in tools:
        if isinstance(tool, dict):
            new_tool = {'type': 'function'}
            function_data = {}
            
            schema = None
            
            if 'function' in tool:
                func = tool['function']
                function_data['name'] = func.get('name', 'unknown_function')
                function_data['description'] = func.get('description', '')
                schema = func.get('parameters') or func.get('input_schema')
            elif 'input_schema' in tool:
                function_data['name'] = tool.get('name', 'unknown_function')
                function_data['description'] = tool.get('description', '')
                schema = tool.get('input_schema')
            elif 'parameters' in tool:
                function_data['name'] = tool.get('name', 'unknown_function')
                function_data['description'] = tool.get('description', '')
                schema = tool.get('parameters')
            elif 'name' in tool or 'description' in tool:
                function_data['name'] = tool.get('name', 'unknown_function')
                function_data['description'] = tool.get('description', '')
                if 'custom' in tool and isinstance(tool['custom'], dict):
                    schema = tool['custom'].get('input_schema')
            else:
                function_data['name'] = 'unknown_function'
                function_data['description'] = ''
            
            if not isinstance(schema, dict):
                schema = {'type': 'object', 'properties': {}}
            if 'type' not in schema:
                schema['type'] = 'object'
            if 'properties' not in schema:
                schema['properties'] = {}
            
            function_data['parameters'] = schema
            new_tool['function'] = function_data
            transformed.append(new_tool)
        else:
            transformed.append(tool)
    return transformed

def validate_max_tokens(data):
    """Validate and fix max_tokens to ensure it's >= 1."""
    if not data:
        return
    
    max_tokens = data.get('max_tokens')
    if not max_tokens or max_tokens == 0:
        data['max_tokens'] = 1024
        return
    
    try:
        max_tokens_int = int(float(max_tokens))
        data['max_tokens'] = 1024 if max_tokens_int < 1 else max_tokens_int
    except (ValueError, TypeError):
        data['max_tokens'] = 1024

def filter_response_headers(headers):
    """Filter out headers that shouldn't be forwarded."""
    excluded = {'content-encoding', 'transfer-encoding', 'connection', 'content-length'}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}

@app.route('/v1/<path:path>', methods=['GET', 'POST'])
def proxy(path):
    clean_path = path.lstrip('/')
    backend_url = f"{BACKEND_BASE_URL}/{clean_path}"
    
    headers = {}
    for k, v in request.headers.items():
        if k.lower() != 'host':
            headers[k] = v
    
    try:
        stream = False
        if request.method == 'GET':
            response = requests.get(backend_url, headers=headers, timeout=30)
        else:
            data = request.get_json()
            if 'Content-Type' not in headers:
                headers['Content-Type'] = 'application/json'
            
            if clean_path == 'chat/completions':
                if data and data.get('model'):
                    incoming_model = str(data.get('model')).lower().replace(' ', '-').replace('--', '-')
                    normalized_rewrite = REWROTE_MODEL_ID.lower().replace(' ', '-').replace('--', '-')
                    if incoming_model == normalized_rewrite:
                        data['model'] = ORIGINAL_MODEL_ID
                
                if data and 'messages' in data:
                    data['messages'] = transform_messages(data['messages'])
                if data and 'tool_choice' in data:
                    data['tool_choice'] = transform_tool_choice(data['tool_choice'])
                if data and 'tools' in data:
                    data['tools'] = transform_tools(data['tools'])
                if data:
                    validate_max_tokens(data)
            
            stream = data.get('stream', False) if data else False
            if stream:
                response = requests.post(backend_url, headers=headers, json=data, timeout=30, stream=True)
            else:
                response = requests.post(backend_url, headers=headers, json=data, timeout=30)
        
        if not stream and response.status_code == 200 and len(response.content or []) == 0:
            try:
                if not response.text or response.text.strip() == '':
                    return jsonify({'error': 'Empty response from provider'}), 500
            except:
                return jsonify({'error': 'Empty response from provider'}), 500
        
        if clean_path == 'models' and response.status_code == 200:
            try:
                models_data = response.json()
                for model in models_data.get('data', []):
                    if model['id'] == ORIGINAL_MODEL_ID:
                        model['id'] = REWROTE_MODEL_ID
                return jsonify(models_data)
            except (ValueError, KeyError):
                pass
        
        if stream and response.status_code == 200:
            def generate():
                try:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                except Exception:
                    yield b''
            
            response_headers = filter_response_headers(response.headers)
            content_type = response.headers.get('Content-Type', 'text/event-stream')
            return Response(generate(), mimetype=content_type, status=response.status_code, headers=response_headers)
        
        response_headers = filter_response_headers(response.headers)
        if 'Content-Type' not in response_headers and response.headers.get('Content-Type'):
            response_headers['Content-Type'] = response.headers.get('Content-Type')
        
        return response.content, response.status_code, response_headers
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=9000, debug=True)
