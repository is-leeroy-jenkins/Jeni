"""Gemini provider wrappers for the Jeni Streamlit application.

Purpose:
    Provides application-facing wrapper classes for Gemini text generation, image generation
    and analysis, embeddings, transcription, translation, text-to-speech, Gemini file
    operations, file-search stores, and Google Cloud Storage bucket workflows.

    The module centralizes provider request construction, option lists, response extraction,
    and error logging so the Streamlit application can call stable Python interfaces instead
    of provider-specific SDK objects directly.

Classes:
    Gemini: Shared base class for Gemini wrapper configuration and runtime state.
    Chat: Text-generation wrapper for Gemini chat and grounding workflows.
    Images: Image generation, analysis, and editing wrapper.
    Embeddings: Text embedding wrapper.
    TTS: Text-to-speech wrapper.
    Transcription: Audio transcription wrapper.
    Translation: Audio translation wrapper.
    Files: Gemini file and document workflow wrapper.
    FileSearch: Gemini file-search store wrapper.
    CloudBuckets: Google Cloud Storage bucket wrapper.

Copyright:
    Copyright © 2026 Terry Eppler.

License:
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the software, subject to inclusion of
    the copyright notice and permission notice in substantial portions of the software.

Contact:
    Terry Eppler, terryeppler@gmail.com or eppler.terry@epa.gov.
"""
from __future__ import annotations
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import json
import os
import requests
import PIL.Image
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from google.cloud.storage.blob import Blob
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore, FileSearch, GenerateContentResponse,
                                GenerateVideosResponse, Image, File, SpeakerVoiceConfig,
                                VoiceConfig, SpeechConfig, Tool, ToolConfig, GoogleSearch,
                                UrlContext, SafetySetting, HarmCategory, HarmBlockThreshold, )

def throw_if( name: str, value: object ) -> None:
	"""Validate that a required argument has a usable value.

	Purpose:
		Provides a shared argument guard for provider calls, file operations, and configuration
		builders. The helper raises a clear ``ValueError`` when a required value is missing,
		blank, or an empty collection.

	Args:
		name (str): Name of the argument being validated.
		value (object): Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is ``None``, blank, or an empty collection.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode a local image file as base64 text.

	Purpose:
		Reads a local image file and converts its binary content into a base64-encoded string.
		This supports workflows that need inline image data instead of a file handle or URI.

	Args:
		image_path (str): Path to the local image file.

	Returns:
		Base64-encoded image content.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( "utf-8" )

class Gemini:
	"""Shared base class for Gemini wrapper configuration and runtime state.

	Purpose:
		Provides common configuration fields used by the Gemini wrapper classes. The base class
		centralizes API keys, model settings, sampling options, modality settings, tool settings,
		and response placeholders used by chat, image, embedding, audio, file, file-search, and
		storage workflows.

	Attributes:
		number (Optional[int]): Candidate or response-count setting.
		google_api_key (Optional[str]): Google API key loaded from configuration.
		gemini_api_key (Optional[str]): Gemini API key loaded from configuration.
		instructions (Optional[str]): System instruction text for model requests.
		prompt (Optional[str]): Prompt text for model requests.
		model (Optional[str]): Active Gemini model name.
		api_version (Optional[str]): Optional API version setting.
		max_tokens (Optional[int]): Maximum output-token setting.
		temperature (Optional[float]): Sampling temperature.
		top_p (Optional[float]): Top-p sampling value.
		top_k (Optional[int]): Top-k sampling value.
		candidate_count (Optional[int]): Candidate count for supported requests.
		media_resolution (Optional[str]): Requested media-resolution option.
		response_modalities (Optional[List[str]]): Requested response modalities.
		stops (Optional[List[str]]): Stop sequences.
		domains (Optional[List[str]]): Domain constraints for supported workflows.
		frequency_penalty (Optional[float]): Frequency penalty for supported requests.
		presence_penalty (Optional[float]): Presence penalty for supported requests.
		response_format (Optional[str]): Requested response MIME type.
		content_response (Optional[GenerateContentResponse]): Last content-generation response.
		image_response (Optional[GenerateImagesResponse]): Last image-generation response.
		content_config (Optional[GenerateContentConfig]): Last content-generation config.
		function_config (Optional[FunctionCallingConfig]): Function-calling configuration.
		thought_config (Optional[ThinkingConfig]): Thinking configuration.
		genimg_config (Optional[GenerateImagesConfig]): Image-generation configuration.
		image_config (Optional[ImageConfig]): Image-specific configuration.
		tool_config (Optional[List[types.Tool]]): Tool configuration.
		tool_choice (Optional[str]): Tool-choice mode.
		tools (Optional[List[str]]): Tool names selected by the caller.
	"""
	
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	media_resolution: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	domains: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize the shared Gemini wrapper state.

		Purpose:
			Initializes provider keys, model settings, sampling fields, request placeholders, and
			response placeholders used by child wrapper classes. The constructor performs local
			state assignment only and does not submit provider requests.
		"""
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.response_modalities = [ ]
		self.stops = [ ]
		self.tools = [ ]


class Chat( Gemini ):
	"""Gemini Interactions API text-generation wrapper.

	Purpose:
		Provides the application-facing text-generation contract used by the Jeni Streamlit
		interface. The class translates existing Jeni prompt, history, tool, structured-output,
		reasoning, URL-context, File Search, and streaming values into Gemini Interactions API
		requests while preserving the return values and runtime members consumed by ``app.py``.

	Attributes:
		use_vertex (Optional[bool]): Whether Vertex AI configuration is active.
		http_options (Optional[Dict[str, Any]]): HTTP client configuration.
		client (Optional[genai.Client]): Gemini SDK client.
		storage_client (Optional[storage.Client]): Optional Google Cloud Storage client.
		contents (Optional[List[Dict[str, Any]]]): Interactions API input steps.
		input_steps (List[Dict[str, Any]]): Normalized Interactions API input timeline.
		image_uri (Optional[str]): Optional image URI retained for compatibility.
		audio_uri (Optional[str]): Optional audio URI retained for compatibility.
		file_path (Optional[str]): Optional local file path retained for compatibility.
		files (List[str]): File identifiers retained for compatibility.
		content_block (Optional[str]): Additional application content prepended to the prompt.
		context (List[Dict[str, Any]]): Existing Jeni chat history.
		urls (List[str]): Normalized URL-context values.
		max_urls (int): Maximum number of URL values included in a request.
		response_schema (Optional[Dict[str, Any]]): Parsed structured-output JSON Schema.
		safety_profile (Optional[str]): UI safety-profile value retained for compatibility.
		safety_settings (Optional[List[SafetySetting]]): Legacy safety configuration placeholder.
		file_search_store_names (List[str]): File Search Store resource names.
		interaction (Optional[Any]): Most recent completed Interaction resource.
		interaction_id (Optional[str]): Identifier of the most recent Interaction.
		previous_interaction_id (Optional[str]): Optional server-side conversation predecessor.
		steps (List[Any]): Model-generated steps returned by the Interactions API.
		response (Optional[Any]): Raw response retained for application usage accounting.
		output_text (str): Extracted text returned to the application.
		grounding_sources (List[Dict[str, Any]]): Sources extracted from annotations and tool
		steps.
		generation_config (Dict[str, Any]): Interactions generation configuration.
		interaction_response_format (Optional[List[Dict[str, Any]]]): Output-format definitions.
		tool_objects (List[Dict[str, Any]]): Interactions server-side tool definitions.
		stream (bool): Whether the request uses server-sent event streaming.
		stream_handler (Optional[Callable[[str], None]]): Application text-delta callback.
		store (bool): Whether Google stores the Interaction resource.
	"""
	
	use_vertex: Optional[ bool ]
	http_options: Optional[ Dict[ str, Any ] ]
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ List[ Dict[ str, Any ] ] ]
	input_steps: List[ Dict[ str, Any ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: List[ str ]
	content_block: Optional[ str ]
	context: List[ Dict[ str, Any ] ]
	urls: List[ str ]
	max_urls: int
	response_schema: Optional[ Dict[ str, Any ] ]
	safety_profile: Optional[ str ]
	safety_settings: Optional[ List[ SafetySetting ] ]
	file_search_store_names: List[ str ]
	interaction: Optional[ Any ]
	interaction_id: Optional[ str ]
	previous_interaction_id: Optional[ str ]
	steps: List[ Any ]
	response: Optional[ Any ]
	output_text: str
	grounding_sources: List[ Dict[ str, Any ] ]
	generation_config: Dict[ str, Any ]
	interaction_response_format: Optional[ List[ Dict[ str, Any ] ] ]
	tool_objects: List[ Dict[ str, Any ] ]
	stream: bool
	stream_handler: Optional[ Callable[ [ str ], None ] ]
	store: bool
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize the Gemini Interactions text wrapper.

		Purpose:
			Initializes the complete compatibility state required by the Jeni application and
			the Gemini Interactions API. The constructor performs local state assignment only and
			does not execute a provider request.

		Args:
			model (str): Default Gemini text-generation model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.api_version = 'v1'
		self.client = None
		self.storage_client = None
		self.http_options = { 'api_version': self.api_version }
		self.use_vertex = False
		self.model = model
		self.prompt = None
		self.instructions = None
		self.number = 1
		self.candidate_count = 1
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.stops = [ ]
		self.response_format = None
		self.response_mime_type = None
		self.response_schema = None
		self.response_modalities = [ ]
		self.media_resolution = None
		self.tool_choice = None
		self.tools = [ ]
		self.tool_objects = [ ]
		self.generation_config = { }
		self.interaction_response_format = None
		self.safety_profile = None
		self.safety_settings = None
		self.include_server_side_tool_invocations = True
		self.contents = None
		self.input_steps = [ ]
		self.content_block = None
		self.context = [ ]
		self.urls = [ ]
		self.max_urls = 0
		self.files = [ ]
		self.file_search_store_names = [ ]
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.interaction = None
		self.interaction_id = None
		self.previous_interaction_id = None
		self.steps = [ ]
		self.response = None
		self.content_response = None
		self.image_response = None
		self.output_text = ''
		self.grounding_metadata = None
		self.grounding_sources = [ ]
		self.stream = False
		self.stream_handler = None
		self.store = False
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini Interactions text models.

		Purpose:
			Provides the model identifiers displayed by the Jeni Text, Document Q&A, File Search
			Stores, and Google Cloud Buckets controls. Existing model identifiers are retained for
			compatibility while currently supported Interactions models are added.

		Returns:
			List[str]: Supported Gemini text-generation model identifiers.
		"""
		return [ 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite',
			'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported Interactions server-side tool names.

		Purpose:
			Provides the tool identifiers consumed by the Jeni Text mode controls and translated
			by ``build_tools`` into Interactions API tool definitions.

		Returns:
			List[str]: Supported server-side tool identifiers.
		"""
		return [ 'google_search', 'google_maps', 'url_context', 'file_search', 'code_execution', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported Interactions thinking levels.

		Purpose:
			Provides the reasoning-level values displayed by Jeni and accepted by the
			Interactions API generation configuration.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Purpose:
			Retains the media-resolution options used by the existing Jeni controls. Text-only
			Interactions requests preserve this value without sending it as an unsupported
			top-level parameter.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported Interactions tool-choice modes.

		Purpose:
			Provides values accepted by the Interactions API ``tool_choice`` parameter.

		Returns:
			List[str]: Supported tool-choice modes.
		"""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return legacy include options retained for UI compatibility.

		Purpose:
			Preserves the option-list contract currently exposed by the Jeni wrapper. Interactions
			tool calls and citations are returned through typed steps and content annotations
			instead of the legacy include mechanism.

		Returns:
			List[str]: Existing application include-option identifiers.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported output-modality values.

		Purpose:
			Provides modality names used to construct Interactions ``response_format`` entries.

		Returns:
			List[str]: Supported output modalities.
		"""
		return [ '', 'text', 'image', 'audio' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported text response MIME types.

		Purpose:
			Provides MIME types used by Jeni's response-format control and structured-output
			configuration.

		Returns:
			List[str]: Supported text response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum' ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Return tools supported by the selected model.

		Purpose:
			Builds the model-specific tool list consumed by the Jeni interface while preserving
			the existing helper contract.

		Args:
			model (str): Gemini model identifier.

		Returns:
			List[str]: Tool identifiers supported by the selected model.

		Raises:
			Error: Raised when validation or tool-option construction fails.
		"""
		try:
			throw_if( 'model', model )
			self.model_name = str( model ).strip( ).lower( )
			self.options = [ 'google_search', 'url_context', 'file_search', 'code_execution', ]
			if self.supports_google_maps( self.model_name ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tools( self, model: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
		"""Return whether a model supports Google Maps grounding.

		Purpose:
			Centralizes model-specific Google Maps feature gating for the Jeni tool controls.

		Args:
			model (str): Gemini model identifier.

		Returns:
			bool: True when Google Maps may be exposed for the model; otherwise False.

		Raises:
			Error: Raised when validation or model comparison fails.
		"""
		try:
			throw_if( 'model', model )
			self.model_name = str( model ).strip( ).lower( )
			self.maps_models = { 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-pro-preview',
				'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite-preview', 'gemini-3-flash-preview',
				'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', }
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'supports_google_maps( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: List[ str ], max_urls: int = 10 ) -> List[ str ]:
		"""Build the normalized URL-context list.

		Purpose:
			Removes blank URL values, preserves their original order, removes duplicates, and
			applies the caller-provided maximum before the values are appended to the request.

		Args:
			urls (List[str]): Candidate URL values.
			max_urls (int): Maximum number of URLs to retain.

		Returns:
			List[str]: Normalized and limited URL values.

		Raises:
			Error: Raised when URL normalization fails.
		"""
		try:
			self.urls = [ ]
			self.max_urls = max( 0, int( max_urls or 0 ) )
			self.url_values = urls if isinstance( urls, list ) else [ ]
			for url in self.url_values:
				if url is None:
					continue
				
				self.url = str( url ).strip( )
				if not self.url:
					continue
				
				if self.url not in self.urls:
					self.urls.append( self.url )
			
			if self.max_urls > 0:
				self.urls = self.urls[ : self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_urls( self,  **kwargs ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_context( self, context: List[ Dict[ str, Any ] ] ) -> List[ Dict[ str, Any ] ]:
		"""Convert Jeni chat history into Interactions input steps.

		Purpose:
			Translates the existing ``role`` and ``content`` dictionaries stored by Jeni into
			``user_input`` and ``model_output`` steps accepted by stateless Interactions requests.
			Unsupported or empty history entries are ignored without modifying the original list.

		Args:
			context (List[Dict[str, Any]]): Existing Jeni chat-history records.

		Returns:
			List[Dict[str, Any]]: Interactions-compatible history steps.

		Raises:
			Error: Raised when context normalization fails.
		"""
		try:
			self.context = context if isinstance( context, list ) else [ ]
			self.history_steps: List[ Dict[ str, Any ] ] = [ ]
			for message in self.context:
				if not isinstance( message, dict ):
					continue
				
				self.role = str( message.get( 'role', '' ) or '' ).strip( ).lower( )
				self.message_content = message.get( 'content', '' )
				if isinstance( self.message_content, list ):
					self.message_text = '\n'.join(
						str( value ).strip( ) for value in self.message_content if
						value is not None and str( value ).strip( ) )
				else:
					self.message_text = str( self.message_content or '' ).strip( )
				
				if not self.message_text:
					continue
				
				if self.role in ('assistant', 'model'):
					self.step_type = 'model_output'
				else:
					self.step_type = 'user_input'
				
				self.history_steps.append( { 'type': self.step_type,
					'content': [ { 'type': 'text', 'text': self.message_text, }, ], } )
			
			return self.history_steps
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'normalize_context( self, **kwargs) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, content: str, context: List[ Dict[ str, Any ] ],
		urls: List[ str ], max_urls: int ) -> List[ Dict[ str, Any ] ]:
		"""Build the complete stateless Interactions input timeline.

		Purpose:
			Combines normalized Jeni history with the current application content, URL references,
			and user prompt. The resulting list preserves conversation order and appends exactly
			one current ``user_input`` step.

		Args:
			prompt (str): Current user prompt.
			content (str): Optional application content prepended to the current prompt.
			context (List[Dict[str, Any]]): Existing Jeni chat history.
			urls (List[str]): Candidate URL-context values.
			max_urls (int): Maximum URL count.

		Returns:
			List[Dict[str, Any]]: Complete Interactions input-step list.

		Raises:
			Error: Raised when input construction fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = str( prompt ).strip( )
			self.content_block = str( content or '' ).strip( )
			self.input_steps = self.normalize_context( context )
			self.urls = self.build_urls( urls, max_urls )
			self.current_parts: List[ str ] = [ ]
			if self.content_block:
				self.current_parts.append( self.content_block )
			
			if self.urls:
				self.current_parts.append(
					'Reference URLs:\n' + '\n'.join( f'- {url}' for url in self.urls ) )
			
			self.current_parts.append( self.prompt )
			self.current_text = '\n\n'.join( self.current_parts ).strip( )
			self.input_steps.append( { 'type': 'user_input',
				'content': [ { 'type': 'text', 'text': self.current_text, }, ], } )
			
			self.contents = self.input_steps
			return self.input_steps
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, **kwargs) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float, top_p: float, top_k: int,
		max_tokens: int, stops: List[ str ], reasoning: str ) -> Dict[ str, Any ]:
		"""Build the Interactions generation configuration.

		Purpose:
			Converts Jeni inference controls into the generation fields supported by Gemini
			Interactions. Zero-valued application defaults are treated as unset values rather
			than overriding model defaults.

		Args:
			temperature (float): Sampling-temperature value.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			max_tokens (int): Maximum output-token count.
			stops (List[str]): Stop sequences.
			reasoning (str): Thinking-level value.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when generation configuration construction fails.
		"""
		try:
			self.temperature = float( temperature or 0.0 )
			self.top_p = float( top_p or 0.0 )
			self.top_k = int( top_k or 0 )
			self.max_tokens = int( max_tokens or 0 )
			self.stops = [ str( value ).strip( ) for value in
				(stops if isinstance( stops, list ) else [ ]) if
				value is not None and str( value ).strip( ) ]
			self.reasoning = str( reasoning or '' ).strip( ).lower( )
			self.generation_config = { }
			if self.temperature > 0.0:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p > 0.0:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.generation_config[ 'top_k' ] = self.top_k
			
			if self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = self.max_tokens
			
			if self.stops:
				self.generation_config[ 'stop_sequences' ] = self.stops
			
			if self.reasoning and self.reasoning not in ('thinking_level_unspecified',
				'unspecified',):
				self.generation_config[ 'thinking_level' ] = self.reasoning
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_generation_config( self, **kwars ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any ) -> Optional[ Dict[ str, Any ] ]:
		"""Parse a structured-output JSON Schema.

		Purpose:
			Accepts the existing Jeni response-schema value as a dictionary or JSON string and
			converts it into the mapping required by the Interactions text response format.

		Args:
			response_schema (Any): Structured-output schema value.

		Returns:
			Optional[Dict[str, Any]]: Parsed JSON Schema, or None when no schema is supplied.

		Raises:
			Error: Raised when a nonblank schema cannot be parsed as a JSON object.
		"""
		try:
			self.response_schema = None
			if response_schema is None:
				return None
			
			if isinstance( response_schema, dict ):
				self.response_schema = response_schema
				return self.response_schema
			
			self.schema_text = str( response_schema ).strip( )
			if not self.schema_text:
				return None
			
			self.schema_value = json.loads( self.schema_text )
			if not isinstance( self.schema_value, dict ):
				raise ValueError( 'The response schema must contain a JSON object.' )
			
			self.response_schema = self.schema_value
			return self.response_schema
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'parse_response_schema( self, **kwargs ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_format: str, response_schema: Any,
		modalities: List[ str ] ) -> List[ Dict[ str, Any ] ]:
		"""Build Interactions output-format definitions.

		Purpose:
			Converts Jeni MIME-type, JSON Schema, and modality controls into the polymorphic
			``response_format`` structure used by the current Interactions API.

		Args:
			response_format (str): Requested text MIME type.
			response_schema (Any): Optional JSON Schema dictionary or JSON string.
			modalities (List[str]): Requested output modalities.

		Returns:
			List[Dict[str, Any]]: Interactions response-format entries.

		Raises:
			Error: Raised when response-format construction fails.
		"""
		try:
			self.response_format = str( response_format or '' ).strip( )
			self.response_mime_type = self.response_format or 'text/plain'
			self.response_schema = self.parse_response_schema( response_schema )
			self.response_modalities = [ str( value ).strip( ).lower( ) for value in
				(modalities if isinstance( modalities, list ) else [ ]) if
				value is not None and str( value ).strip( ) ]
			
			if not self.response_modalities:
				self.response_modalities = [ 'text' ]
			
			self.interaction_response_format = [ ]
			
			if 'text' in self.response_modalities:
				self.text_format: Dict[ str, Any ] = { 'type': 'text',
					'mime_type': self.response_mime_type, }
				
				if self.response_schema is not None:
					self.text_format[ 'mime_type' ] = 'application/json'
					self.text_format[ 'schema' ] = self.response_schema
				
				self.interaction_response_format.append( self.text_format )
			
			if 'image' in self.response_modalities:
				self.interaction_response_format.append( { 'type': 'image' } )
			
			if 'audio' in self.response_modalities:
				self.interaction_response_format.append( { 'type': 'audio' } )
			
			if not self.interaction_response_format:
				self.interaction_response_format.append(
					{ 'type': 'text', 'mime_type': self.response_mime_type, } )
			
			return self.interaction_response_format
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_response_format( self, **kwargs ) -> List[ Dict[ str, Any ]]'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ str ], urls: List[ str ],
		file_search_store_names: List[ str ] ) -> List[ Dict[ str, Any ] ]:
		"""Build Interactions server-side tool definitions.

		Purpose:
			Translates Jeni tool names into Interactions API tool dictionaries. URL Context is
			enabled when selected or when URLs are supplied. File Search is enabled when one or
			more File Search Store resource names are provided.

		Args:
			tools (List[str]): Tool identifiers selected by the caller.
			urls (List[str]): Normalized URL-context values.
			file_search_store_names (List[str]): File Search Store resource names.

		Returns:
			List[Dict[str, Any]]: Interactions server-side tool definitions.

		Raises:
			Error: Raised when tool-definition construction fails.
		"""
		try:
			self.tools = [ str( value ).strip( ).lower( ) for value in
				(tools if isinstance( tools, list ) else [ ]) if
				value is not None and str( value ).strip( ) ]
			self.file_search_store_names = [ str( value ).strip( ) for value in
				(file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
				if value is not None and str( value ).strip( ) ]
			self.tool_objects = [ ]
			
			if 'google_search' in self.tools:
				self.tool_objects.append( { 'type': 'google_search' } )
			
			if 'google_maps' in self.tools and self.supports_google_maps( self.model ):
				self.tool_objects.append( { 'type': 'google_maps' } )
			
			if 'url_context' in self.tools or urls:
				self.tool_objects.append( { 'type': 'url_context' } )
			
			if 'code_execution' in self.tools:
				self.tool_objects.append( { 'type': 'code_execution' } )
			
			if self.file_search_store_names:
				self.tool_objects.append( { 'type': 'file_search',
					'file_search_store_names': self.file_search_store_names, } )
			
			return self.tool_objects
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, **kwargs) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_value( self, value: Any ) -> Any:
		"""Convert an SDK value into standard Python data.

		Purpose:
			Normalizes Pydantic SDK objects, dictionaries, lists, tuples, and scalar values for
			grounding-source extraction without changing provider response state.

		Args:
			value (Any): SDK or Python value to normalize.

		Returns:
			Any: Equivalent standard Python value.
		"""
		if value is None or isinstance( value, (str, int, float, bool) ):
			return value
		
		if isinstance( value, dict ):
			return { key: self.normalize_value( item ) for key, item in value.items( ) }
		
		if isinstance( value, (list, tuple) ):
			return [ self.normalize_value( item ) for item in value ]
		
		if hasattr( value, 'model_dump' ):
			return self.normalize_value( value.model_dump( ) )
		
		return str( value )
	
	def capture_interaction( self, interaction: Any ) -> None:
		"""Capture a completed unary Interaction.

		Purpose:
			Stores the raw Interaction, identifier, steps, output text, and grounding sources on
			the wrapper members consumed by the Jeni application.

		Args:
			interaction (Any): Completed Gemini Interaction resource.

		Returns:
			None: This method updates wrapper state through side effects.

		Raises:
			Error: Raised when response-state extraction fails.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.response = interaction
			self.content_response = interaction
			self.interaction_id = getattr( interaction, 'id', None )
			self.steps = list( getattr( interaction, 'steps', None ) or [ ] )
			self.output_text = str( getattr( interaction, 'output_text', '' ) or '' ).strip( )
			self.grounding_sources = self.extract_grounding_sources( interaction )
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'capture_interaction( self, interaction: Any ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def extract_grounding_sources( self, interaction: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract grounding citations from Interaction steps.

		Purpose:
			Collects URL citations, File Search citations, and search-result metadata from
			``model_output`` annotations and server-side tool-result steps. Duplicate sources are
			removed while preserving provider order.

		Args:
			interaction (Any): Gemini Interaction containing model-generated steps.

		Returns:
			List[Dict[str, Any]]: Normalized grounding-source records.

		Raises:
			Error: Raised when grounding-source extraction fails.
		"""
		try:
			self.source_values: List[ Dict[ str, Any ] ] = [ ]
			self.source_keys: set[ tuple[ str, str, str ] ] = set( )
			self.interaction_steps = getattr( interaction, 'steps', None ) or [ ]
			for step in self.interaction_steps:
				self.step_type = str( getattr( step, 'type', '' ) or '' ).strip( )
				
				if self.step_type == 'model_output':
					self.step_content = getattr( step, 'content', None ) or [ ]
					for block in self.step_content:
						self.annotations = getattr( block, 'annotations', None ) or [ ]
						for annotation in self.annotations:
							self.annotation_value = self.normalize_value( annotation )
							if not isinstance( self.annotation_value, dict ):
								continue
							
							self.append_source( source=self.annotation_value, default_type=str(
								self.annotation_value.get( 'type', 'citation' ) ), )
				
				elif self.step_type in ('google_search_result', 'file_search_result',
					'url_context_result', 'google_maps_result',):
					self.result_value = self.normalize_value( getattr( step, 'result', None ) )
					
					if isinstance( self.result_value, list ):
						for source in self.result_value:
							if isinstance( source, dict ):
								self.append_source( source=source, default_type=self.step_type, )
					
					elif isinstance( self.result_value, dict ):
						self.append_source( source=self.result_value,
							default_type=self.step_type, )
			
			return self.source_values
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'extract_grounding_sources( self, **kwargs ) -> List[Dict[str, Any]]'
			Logger( ).write( exception )
			raise exception
	
	def append_source( self, source: Dict[ str, Any ], default_type: str ) -> None:
		"""Append one normalized grounding source.

		Purpose:
			Converts provider-specific annotation and tool-result fields into the stable source
			shape consumed by Jeni and prevents duplicate entries.

		Args:
			source (Dict[str, Any]): Provider source or annotation mapping.
			default_type (str): Source type used when the mapping omits one.

		Returns:
			None: This method updates ``grounding_sources`` construction state.
		"""
		self.source_type = str( source.get( 'type', default_type ) or default_type ).strip( )
		self.source_title = str(
			source.get( 'title' ) or source.get( 'display_name' ) or source.get(
				'file_name' ) or source.get( 'name' ) or '' ).strip( )
		self.source_url = str( source.get( 'url' ) or source.get( 'uri' ) or source.get(
			'source_url' ) or '' ).strip( )
		self.source_text = str(
			source.get( 'text' ) or source.get( 'snippet' ) or source.get( 'quote' ) or source.get(
				'search_suggestions' ) or '' ).strip( )
		self.source_file_id = str(
			source.get( 'file_id' ) or source.get( 'file_name' ) or source.get(
				'document_name' ) or '' ).strip( )
		
		if not any( (self.source_title, self.source_url, self.source_text, self.source_file_id,) ):
			return
		
		self.source_key = (self.source_type, self.source_url or self.source_file_id,
			self.source_text,)
		
		if self.source_key in self.source_keys:
			return
		
		self.source_keys.add( self.source_key )
		self.source_values.append( { 'type': self.source_type, 'title': self.source_title or None,
			'snippet': self.source_text or None, 'url': self.source_url or None,
			'files_id': self.source_file_id or None, 'metadata': source, } )
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""Return sources from the most recent Interaction.

		Purpose:
			Restores the application-facing source accessor called by Jeni Text mode after a
			grounded Gemini response.

		Returns:
			List[Dict[str, Any]]: Grounding sources extracted from the latest response.
		"""
		return list( self.grounding_sources )
	
	def generate_text_stream( self ) -> str:
		"""Execute a streaming Interactions text request.

		Purpose:
			Submits the validated request using Interactions server-sent events, forwards each text
			delta to the existing Jeni callback, accumulates the complete answer, and retains the
			final Interaction metadata used by token accounting.

		Returns:
			str: Complete generated text accumulated from stream deltas.

		Raises:
			Error: Raised when streaming execution or event handling fails.
		"""
		try:
			self.stream_response = self.client.interactions.create( model=self.model,
				input=self.input_steps, system_instruction=self.instructions or None,
				tools=self.tool_objects or None, generation_config=self.generation_config or None,
				response_format=self.interaction_response_format,
				tool_choice=self.tool_choice or None, stream=True, store=self.store, )
			
			self.text_chunks: List[ str ] = [ ]
			self.streaming_interaction = None
			for event in self.stream_response:
				self.event_type = str( getattr( event, 'event_type', '' ) or '' ).strip( )
				if self.event_type == 'step.delta':
					self.delta = getattr( event, 'delta', None )
					self.delta_type = str( getattr( self.delta, 'type', '' ) or '' ).strip( )
					if self.delta_type == 'text':
						self.delta_text = str( getattr( self.delta, 'text', '' ) or '' )
						if self.delta_text:
							self.text_chunks.append( self.delta_text )
							if callable( self.stream_handler ):
								self.stream_handler( self.delta_text )
				
				elif self.event_type == 'interaction.completed':
					self.streaming_interaction = getattr( event, 'interaction', None, )
				
				elif self.event_type == 'error':
					self.stream_error = getattr( event, 'error', None )
					self.stream_message = str( getattr( self.stream_error, 'message',
						'' ) or self.stream_error or 'Gemini streaming request failed.' )
					raise RuntimeError( self.stream_message )
			
			self.output_text = ''.join( self.text_chunks ).strip( )
			if self.streaming_interaction is not None:
				self.interaction = self.streaming_interaction
				self.response = self.streaming_interaction
				self.content_response = self.streaming_interaction
				self.interaction_id = getattr( self.streaming_interaction, 'id', None, )
				self.steps = list( getattr( self.streaming_interaction, 'steps', None ) or [ ] )
			else:
				self.response = self.stream_response
				self.content_response = self.stream_response
			
			self.grounding_sources = [ ]
			return self.output_text
		
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text_stream( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite', number: int = 1,
		temperature: float = 0.0, top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0,
		presence: float = 0.0, max_tokens: int = 0, stops: Optional[ List[ str ] ] = None,
		instruct: str = '', response_format: str = '', tools: Optional[ List[ str ] ] = None,
		tool_choice: Optional[ str ] = None, reasoning: str = '',
		modalities: Optional[ List[ str ] ] = None, media_resolution: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, content: str = '',
		urls: Optional[ List[ str ] ] = None, max_urls: int = 0, response_schema: Any = '',
		safety_profile: str = '', file_search_store_names: Optional[ List[ str ] ] = None,
		stream: bool = False, stream_handler: Optional[ Callable[ [ str ], None ] ] = None ) -> (
			str):
		"""Generate text through the Gemini Interactions API.

		Purpose:
			Preserves the complete ``generate_text`` contract currently called by the Jeni
			application. The method validates and assigns every argument to wrapper state, builds
			stateless Interactions input and configuration objects, executes unary or streaming
			generation, retains the raw response, and returns normalized text.

		Args:
			prompt (str): Current user prompt.
			model (str): Gemini model identifier.
			number (int): Candidate-count value retained for application compatibility.
			temperature (float): Sampling-temperature value.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency-penalty value retained for compatibility.
			presence (float): Presence-penalty value retained for compatibility.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			instruct (str): System instruction text.
			response_format (str): Requested text MIME type.
			tools (Optional[List[str]]): Server-side tool identifiers.
			tool_choice (Optional[str]): Tool-selection mode.
			reasoning (str): Thinking-level value.
			modalities (Optional[List[str]]): Requested output modalities.
			media_resolution (str): Media-resolution value retained for compatibility.
			context (Optional[List[Dict[str, Any]]]): Existing Jeni chat history.
			content (str): Additional application content.
			urls (Optional[List[str]]): URL-context values.
			max_urls (int): Maximum number of URLs.
			response_schema (Any): Optional JSON Schema mapping or JSON string.
			safety_profile (str): Safety-profile value retained for compatibility.
			file_search_store_names (Optional[List[str]]): File Search Store resource names.
			stream (bool): Whether to stream text deltas.
			stream_handler (Optional[Callable[[str], None]]): Text-delta callback.

		Returns:
			str: Generated response text.

		Raises:
			Error: Raised when validation, request construction, or provider execution fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = str( prompt ).strip( )
			self.model = str( model ).strip( )
			self.number = max( 1, int( number or 1 ) )
			self.candidate_count = self.number
			self.temperature = float( temperature or 0.0 )
			self.top_p = float( top_p or 0.0 )
			self.top_k = int( top_k or 0 )
			self.frequency_penalty = float( frequency or 0.0 )
			self.presence_penalty = float( presence or 0.0 )
			self.max_tokens = int( max_tokens or 0 )
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = str( instruct or '' ).strip( )
			self.response_format = str( response_format or '' ).strip( )
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.tool_choice = str( tool_choice or '' ).strip( ).lower( ) or None
			self.reasoning = str( reasoning or '' ).strip( )
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.media_resolution = str( media_resolution or '' ).strip( )
			self.context = context if isinstance( context, list ) else [ ]
			self.content_block = str( content or '' ).strip( )
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max( 0, int( max_urls or 0 ) )
			self.safety_profile = str( safety_profile or '' ).strip( )
			self.file_search_store_names = (
				file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
			self.stream = bool( stream )
			self.stream_handler = stream_handler
			self.store = False
			self.previous_interaction_id = None
			self.gemini_api_key =cfg.GEMINI_API_KEY or cfg.GOOGLE_API_KEY
			self.client = genai.Client( api_key=self.gemini_api_key,
				http_options=types.HttpOptions( api_version=self.api_version ), )
			self.input_steps = self.build_input( prompt=self.prompt, content=self.content_block,
				context=self.context, urls=self.urls, max_urls=self.max_urls, )
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, top_k=self.top_k, max_tokens=self.max_tokens, stops=self.stops,
				reasoning=self.reasoning, )
			self.interaction_response_format = self.build_response_format(
				response_format=self.response_format, response_schema=response_schema,
				modalities=self.response_modalities, )
			self.tool_objects = self.build_tools( tools=self.tools, urls=self.urls,
				file_search_store_names=self.file_search_store_names, )
			
			if self.stream:
				return self.generate_text_stream( )
			
			self.interaction = self.client.interactions.create( model=self.model,
				input=self.input_steps, system_instruction=self.instructions or None,
				tools=self.tool_objects or None, generation_config=self.generation_config or None,
				response_format=self.interaction_response_format,
				tool_choice=self.tool_choice or None, stream=False, store=self.store, )
			
			self.capture_interaction( self.interaction )
			if not self.output_text:
				raise ValueError( 'Gemini returned an empty Interactions response.' )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, **Kwargs ) -> str'
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Images class.
	
	Purpose:
		Supports Gemini image generation, image analysis, and image editing workflows. The class
		stores image-specific configuration, constructs multimodal request payloads, manages
		grounding options when supported, and extracts image or text output from provider
		responses.
	
	Attributes:
		client (Optional[genai.Client]): Runtime field used by the Images workflow.
		aspect_ratio (Optional[str]): Runtime field used by the Images workflow.
		use_vertex (Optional[bool]): Runtime field used by the Images workflow.
		resolution (Optional[str]): Runtime field used by the Images workflow.
		size (Optional[str]): Runtime field used by the Images workflow.
	"""
	
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	resolution: Optional[ str ]
	size: Optional[ str ]
	
	def __init__( self, model: str='gemini-2.5-flash-image' ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Images instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			model (str): model value used by this workflow.
		"""
		super( ).__init__( )
		self.number = None
		self.model = model
		self.client = None
		self.instructions = None
		self.image_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_config = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.stops = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.temperature = None
		self.size = None
		self.top_p = None
		self.top_k = None
		self.aspect_ratio = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_output_tokens = None
		self.use_vertex = None
		self.media_resolution = None
		self.tool_choice = None
		self.content_response = None
		self.response = None
		self.grounding_metadata = None
		self.output_mime_type = None
		self.response_mode = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-2.5-flash-image', 'gemini-3.1-flash-image-preview' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
			Returns the include options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Aspect options.
		
		Purpose:
			Returns the aspect options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.
		
		Purpose:
			Returns the media options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
			Returns the modality options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'text', 'image', 'text_and_image' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
			Returns the reasoning options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'unspecified', 'minimal', 'low', 'medium', 'high' ]
	
	@property
	def size_options( self ) -> List[ str ] | None:
		"""Size options.
		
		Purpose:
			Returns the size options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		"""
		return [ '1K', '2K', '4K' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
			Returns the tool options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'google_search', 'image_search' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
			Returns the choice options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
			Returns the format options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
			Returns the mime options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'image/jpeg', 'image/png', 'image/webp' ]
	
	@property
	def resolution_options( self ) -> List[ str ] | None:
		"""Resolution options.
		
		Purpose:
			Returns the resolution options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ '1K', '2K', '4K' ]
	
	def supports_image_size( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""Supports image size.
		
		Purpose:
			Determines whether the selected model supports a specific Images feature. The method
			centralizes feature gating so the UI and request builders expose only compatible
			options.
		
		Args:
			model (str): model value used by this workflow.
		
		Returns:
			True when the selected model supports the requested feature; otherwise False.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.image_size_models = [ 'gemini-3.1-flash-image-preview',
				'gemini-3-pro-image-preview' ]
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""Supports search grounding.
		
		Purpose:
			Determines whether the selected model supports a specific Images feature. The method
			centralizes feature gating so the UI and request builders expose only compatible
			options.
		
		Args:
			model (str): model value used by this workflow.
		
		Returns:
			True when the selected model supports the requested feature; otherwise False.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.search_grounding_models = [ 'gemini-3.1-flash-image-preview',
				'gemini-3-pro-image-preview' ]
			return self.model_name in self.search_grounding_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_search_grounding( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""Supports image search.
		
		Purpose:
			Determines whether the selected model supports a specific Images feature. The method
			centralizes feature gating so the UI and request builders expose only compatible
			options.
		
		Args:
			model (str): model value used by this workflow.
		
		Returns:
			True when the selected model supports the requested feature; otherwise False.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
		image_only: bool=False ) -> List[ str ]:
		"""Normalize response modalities.
		
		Purpose:
			Normalizes input values for the Images workflow before they are passed to provider 
			calls
			or downstream processing. The method converts UI or caller-supplied values into a 
			stable
			shape expected by the wrapper.
		
		Args:
			response_modalities (Optional[str]): response modalities value used by this workflow.
			image_only (bool): image only value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.mode_name = str( response_modalities or '' ).strip( ).upper( )
			if self.mode_name == 'TEXT_AND_IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT':
				return [ 'TEXT' ]
			
			if self.mode_name == 'IMAGE':
				return [ 'IMAGE' ]
			
			if self.mode_name == 'TEXT,IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT, IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			return [ 'IMAGE' ] if image_only else [ 'TEXT' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'normalize_response_modalities( self, *args ) -> List[str]'
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tool( self, image_search: bool=False ) -> Optional[ Tool ]:
		"""Build grounding tool.
		
		Purpose:
			Builds the request component used by the Images workflow. The method translates caller
			options and current object state into provider-compatible configuration or content
			values.
		
		Args:
			image_search (bool): image search value used by this workflow.
		
		Returns:
			Provider-compatible request component or configuration value.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			if not self.supports_search_grounding( self.model ):
				return None
			
			self.use_image_search = bool( image_search )
			self.model_name = str( self.model or '' ).strip( ).lower( )
			if self.use_image_search and self.supports_image_search( self.model_name ):
				return Tool( google_search=types.GoogleSearch(
					search_types=types.SearchTypes( web_search=types.WebSearch( ),
						image_search=types.ImageSearch( ) ) ) )
			
			return Tool( google_search=types.GoogleSearch( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_grounding_tool( self, *args ) -> Optional[Tool]'
			Logger( ).write( exception )
			raise exception
	
	def get_content_config( self, response_modalities: Optional[ str ], image_only: bool=False,
		image_search: bool=False, grounded: bool=False, 
		output_mime_type: Optional[ str ]=None ) -> GenerateContentConfig:
		"""Get content config.
		
		Purpose:
			Retrieves a derived value from the current Images runtime state. The method shields
			callers from provider response-shape differences and returns a stable 
			application-facing
			value.
		
		Args:
			response_modalities (Optional[str]): response modalities value used by this workflow.
			image_only (bool): image only value used by this workflow.
			image_search (bool): image search value used by this workflow.
			grounded (bool): grounded value used by this workflow.
			output_mime_type (Optional[str]): output mime type value used by this workflow.
		
		Returns:
			Derived value extracted from the current runtime state.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.image_only = image_only
			self.image_config = None
			self.tool_config = None
			self.grounding_metadata = None
			self.output_mime_type = str( output_mime_type or '' ).strip( ) or None
			self.image_kwargs = { }
			self.aspect_value = str( self.aspect_ratio or '' ).strip( )
			if self.aspect_value:
				self.image_kwargs[ 'aspect_ratio' ] = self.aspect_value
			
			self.size_value = str( self.size or '' ).strip( )
			if self.size_value and self.supports_image_size( self.model ):
				self.image_kwargs[ 'image_size' ] = self.size_value
			
			if len( self.image_kwargs ) > 0:
				self.image_config = types.ImageConfig( **self.image_kwargs )
			
			if grounded:
				self.grounding_tool = self.build_grounding_tool( image_search=image_search )
				if self.grounding_tool is not None:
					self.tool_config = [ self.grounding_tool ]
			
			self.response_modalities = self.normalize_response_modalities(
				response_modalities=response_modalities, image_only=image_only )
			
			self.config_kwargs = { 'response_modalities': self.response_modalities }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.number is not None and int( self.number or 0 ) > 0:
				self.config_kwargs[ 'candidate_count' ] = int( self.number )
			
			if self.max_output_tokens is not None and int( self.max_output_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_output_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.image_config is not None:
				self.config_kwargs[ 'image_config' ] = self.image_config
			
			if self.tool_config is not None and len( self.tool_config ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_content_config( self, **kwargs ) -> GenerateContentConfig'
			Logger( ).write( exception )
			raise exception
	
	def open_image( self, path: str ) -> PIL.Image.Image:
		"""Open image.
		
		Purpose:
			Opens a local image file and returns an independent image object for multimodal
			requests. The method validates the path and copies the image so the file handle can
			close safely.
		
		Args:
			path (str): path value used by this workflow.
		
		Returns:
			Result produced by the operation.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'path', path )
			with PIL.Image.open( path ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'open_image( self, path ) -> PIL.Image.Image'
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture metadata.
		
		Purpose:
			Captures response metadata from the most recent Images provider response. The method
			stores provider metadata on the instance for later display, citation, grounding, or
			diagnostic use.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.grounding_metadata = None
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				for candidate in self.candidates:
					self.metadata = getattr( candidate, 'grounding_metadata', None )
					if self.metadata is None:
						self.metadata = getattr( candidate, 'groundingMetadata', None )
					
					if self.metadata is not None:
						self.grounding_metadata = self.metadata
						return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'capture_metadata( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Get first image.
		
		Purpose:
			Retrieves a derived value from the current Images runtime state. The method shields
			callers from provider response-shape differences and returns a stable 
			application-facing
			value.
		
		Returns:
			Derived value extracted from the current runtime state.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			if self.content_response is None:
				return None
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				for part in parts:
					try:
						if getattr( part, 'inline_data', None ) is not None:
							return part.as_image( )
					except Exception as e:
						exception = Error( e )
						exception.module = 'gemini'
						exception.cause = 'Images'
						exception.method = 'get_first_image( self ) -> Optional[ PIL.Image.Image ]'
						Logger( ).write( exception )
						continue
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					candidate_parts = getattr( content, 'parts', None ) or [ ]
					for part in candidate_parts:
						try:
							if getattr( part, 'inline_data', None ) is not None:
								return part.as_image( )
						except Exception as e:
							exception = Error( e )
							exception.module = 'gemini'
							exception.cause = 'Images'
							exception.method = ('get_first_image( self ) -> Optional[ Image ]')
							Logger( ).write( exception )
							continue
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_first_image( self ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Get output text.
		
		Purpose:
			Retrieves a derived value from the current Images runtime state. The method shields
			callers from provider response-shape differences and returns a stable 
			application-facing
			value.
		
		Returns:
			Derived value extracted from the current runtime state.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			if self.content_response is None:
				return None
			
			text = getattr( self.content_response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				return text
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				output = [ ]
				for part in parts:
					part_text = getattr( part, 'text', None )
					if isinstance( part_text, str ) and part_text.strip( ):
						output.append( part_text.strip( ) )
				
				if output:
					return '\n'.join( output )
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					output = [ ]
					for part in getattr( content, 'parts', None ) or [ ]:
						part_text = getattr( part, 'text', None )
						if isinstance( part_text, str ) and part_text.strip( ):
							output.append( part_text.strip( ) )
					
					if output:
						return '\n'.join( output )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_output_text( self ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str='gemini-2.5-flash-image', aspect: str=None,
		number: int=None, temperature: float=None, top_p: float=None, frequency: float= None,
		presence: float=None, max_tokens: int=None, resolution: str=None,
		instruct: str=None, output_mime_type: str=None, response_modalities: str=None,
		grounded: bool=False, image_search: bool=False ) -> Optional[ PIL.Image.Image ]:
		"""Generate.
		
		Purpose:
			Executes the generate workflow for the Images wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			model (str): model value used by this workflow.
			aspect (str): aspect value used by this workflow.
			number (int): number value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			resolution (str): resolution value used by this workflow.
			instruct (str): instruct value used by this workflow.
			output_mime_type (str): output mime type value used by this workflow.
			response_modalities (str): response modalities value used by this workflow.
			grounded (bool): grounded value used by this workflow.
			image_search (bool): image search value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config( image_only=True, grounded=grounded,
				image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, aspect ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str='gemini-2.5-flash-image',
		aspect: str=None, number: int=None, temperature: float=None, top_p: float=None,
		frequency: float=None, presence: float=None, max_tokens: int=None,
		resolution: str=None, instruct: str=None, output_mime_type: str=None,
		response_modalities: str=None, grounded: bool=False,
		image_search: bool=False ) -> Optional[ str ]:
		"""Analyze.
		
		Purpose:
			Executes the analyze workflow for the Images wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			path (str): path value used by this workflow.
			model (str): model value used by this workflow.
			aspect (str): aspect value used by this workflow.
			number (int): number value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			resolution (str): resolution value used by this workflow.
			instruct (str): instruct value used by this workflow.
			output_mime_type (str): output mime type value used by this workflow.
			response_modalities (str): response modalities value used by this workflow.
			grounded (bool): grounded value used by this workflow.
			image_search (bool): image search value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.media_resolution = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities or 'text'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config( image_only=False, grounded=grounded,
				image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.open_image( path ) ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, prompt, path, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str='gemini-2.5-flash-image',
		aspect: str=None, number: int=None, temperature: float=None, top_p: float=None,
		frequency: float=None, presence: float=None, max_tokens: int=None,
		resolution: str=None, instruct: str=None, output_mime_type: str=None,
		response_modalities: str=None, grounded: bool=False,
		image_search: bool=False ) ->  Optional[ PIL.Image.Image ]:
		"""Edit.
		
		Purpose:
			Executes the edit workflow for the Images wrapper. The method validates required 
			inputs,
			prepares provider configuration, performs the requested provider or storage operation,
			captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			path (str): path value used by this workflow.
			model (str): model value used by this workflow.
			aspect (str): aspect value used by this workflow.
			number (int): number value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			resolution (str): resolution value used by this workflow.
			instruct (str): instruct value used by this workflow.
			output_mime_type (str): output mime type value used by this workflow.
			response_modalities (str): response modalities value used by this workflow.
			grounded (bool): grounded value used by this workflow.
			image_search (bool): image search value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities or 'image'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self.get_content_config( image_only=True, grounded=grounded,
				image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.open_image( path ) ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt, path, model ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception

class Embeddings( Gemini ):
	"""Embeddings class.
	
	Purpose:
		Creates vector embeddings from text input using Gemini embedding models. The class
		normalizes single-string and batch text input, builds embedding configuration, calls the
		provider API, and extracts vector values in a stable application-facing shape.
	
	Attributes:
		client (Optional[genai.Client]): Runtime field used by the Embeddings workflow.
		response (Optional[Any]): Runtime field used by the Embeddings workflow.
		embedding (Optional[List[float] | List[List[float]]]): Runtime field used by the 
		Embeddings workflow.
		encoding_format (Optional[str]): Runtime field used by the Embeddings workflow.
		dimensions (Optional[int]): Runtime field used by the Embeddings workflow.
		task_type (Optional[str]): Runtime field used by the Embeddings workflow.
		title (Optional[str]): Runtime field used by the Embeddings workflow.
		embedding_config (Optional[types.EmbedContentConfig]): Runtime field used by the 
		Embeddings workflow.
		contents (Optional[str | List[str]]): Runtime field used by the Embeddings workflow.
		input_text (Optional[str | List[str]]): Runtime field used by the Embeddings workflow.
		file_path (Optional[str]): Runtime field used by the Embeddings workflow.
		response_modalities (Optional[str]): Runtime field used by the Embeddings workflow.
	"""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] | List[ List[ float ] ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	title: Optional[ str ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ str | List[ str ] ]
	input_text: Optional[ str | List[ str ] ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str='gemini-embedding-001' ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Embeddings instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			model (str): model value used by this workflow.
		"""
		super( ).__init__( )
		self.model = model
		self.client = None
		self.embedding = None
		self.embeddings = None
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.contents = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.title = None
		self.response_modalities = None
		self.embedding_config = None
		self.content_config = None
		self.api_key = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-embedding-001', 'gemini-embedding-2', 'gemini-embedding-2-preview',
			'text-embedding-004', 'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Encoding options.
		
		Purpose:
			Returns the encoding options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Task options.
		
		Purpose:
			Returns the task options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ '', 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY',
			'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION',
			'CODE_RETRIEVAL_QUERY' ]
	
	def normalize_dimensions( self, dimensions: int ) -> int | None:
		"""Normalize dimensions.
		
		Purpose:
			Normalizes input values for the Embeddings workflow before they are passed to provider
			calls or downstream processing. The method converts UI or caller-supplied values into a
			stable shape expected by the wrapper.
		
		Args:
			dimensions (int): dimensions value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		"""
		try:
			throw_if( 'dimensions', dimensions )
			self.dimensions = dimensions
			if self.dimensions <= 0:
				return None
			
			return self.dimensions
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'refresh_collections( self ) -> Dict[ str, str ]'
			Logger( ).write( exception )
			return None
	
	def normalize_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Normalize contents.
		
		Purpose:
			Normalizes input values for the Embeddings workflow before they are passed to provider
			calls or downstream processing. The method converts UI or caller-supplied values into a
			stable shape expected by the wrapper.
		
		Args:
			text (str | List[str]): text value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'text', text )
			
			if isinstance( text, list ):
				self.contents = [ ]
				for item in text:
					if item is None:
						continue
					
					self.item = str( item ).strip( )
					if self.item:
						self.contents.append( self.item )
				
				throw_if( 'text', self.contents )
				return self.contents
			
			self.contents = str( text ).strip( )
			throw_if( 'text', self.contents )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'normalize_contents( self, text: str | List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def build_embedding_config( self, model: str='gemini-embedding-001', dimensions: int=None,
		task_type: str=None, title: str=None ) -> EmbedContentConfig:
		"""Build embedding config.
		
		Purpose:
			Builds the request component used by the Embeddings workflow. The method translates
			caller options and current object state into provider-compatible configuration or
			content values.
		
		Args:
			model (str): model value used by this workflow.
			dimensions (int): dimensions value used by this workflow.
			task_type (str): task type value used by this workflow.
			title (str): title value used by this workflow.
		
		Returns:
			Provider-compatible request component or configuration value.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type.strip( ).upper( )
			self.title = title.strip( )
			self.config_kwargs = { }
			
			if self.dimensions is not None:
				self.config_kwargs[ 'output_dimensionality' ] = self.dimensions
			
			if self.task_type and 'gemini-embedding-2' not in self.model:
				self.config_kwargs[ 'task_type' ] = self.task_type
			
			if (self.title and self.task_type == 'RETRIEVAL_DOCUMENT' and 'gemini-embedding-2' not 
					in self.model):
				self.config_kwargs[ 'title' ] = self.title
			
			self.embedding_config = EmbedContentConfig( **self.config_kwargs )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('build_embedding_config( self, model, dimensions, task_type, '
			                    'title )')
			Logger( ).write( exception )
			raise exception
	
	def extract_embeddings( self ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Extract embeddings.
		
		Purpose:
			Extracts provider response values from the current Embeddings response object. The
			method normalizes nested SDK response structures into the return shape expected by the
			application.
		
		Returns:
			Result produced by the operation.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			if self.response is None:
				return None
			
			if not hasattr( self.response, 'embeddings' ):
				return None
			
			self.embeddings = [ ]
			for item in self.response.embeddings:
				if item is None:
					continue
				
				if hasattr( item, 'values' ) and item.values is not None:
					self.embeddings.append( list( item.values ) )
			
			if len( self.embeddings ) == 0:
				return None
			
			if len( self.embeddings ) == 1 and isinstance( self.input_text, str ):
				self.embedding = self.embeddings[ 0 ]
				return self.embedding
			
			self.embedding = self.embeddings
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'extract_embeddings( self )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str='gemini-embedding-001',
		dimensions: int=None, task_type: str=None, title: str=None,
		encoding_format: str='float' ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Create.
		
		Purpose:
			Executes the create workflow for the Embeddings wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			text (str | List[str]): text value used by this workflow.
			model (str): model value used by this workflow.
			dimensions (int): dimensions value used by this workflow.
			task_type (str): task type value used by this workflow.
			title (str): title value used by this workflow.
			encoding_format (str): encoding format value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'text', text )
			self.api_key = cfg.GEMINI_API_KEY
			throw_if( 'api_key', self.api_key )
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.encoding_format = encoding_format
			self.input_text = self.normalize_contents( text=text )
			self.model = model.strip( )
			self.encoding_format = encoding_format
			self.embedding_config = self.build_embedding_config( model=self.model,
				dimensions=self.imensions, task_type=self.task_type, title=self.title )
			self.client = genai.Client( api_key=self.api_key )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('create( self, *args ) -> List[ float ] | List[ List[ float ] ]')
			Logger( ).write( exception )
			raise exception

class TTS( Gemini ):
	"""TTS class.
	
	Purpose:
		Converts text input into spoken audio using Gemini text-to-speech models. The class
		normalizes voice and prompt options, requests audio output from the provider, wraps raw
		PCM bytes in a WAV container, and optionally writes the generated audio to disk.
	
	Attributes:
		speed (Optional[float]): Runtime field used by the TTS workflow.
		voice (Optional[str]): Runtime field used by the TTS workflow.
		response (Optional[GenerateContentResponse]): Runtime field used by the TTS workflow.
		voice_config (Optional[VoiceConfig]): Runtime field used by the TTS workflow.
		speech_config (Optional[SpeechConfig]): Runtime field used by the TTS workflow.
		client (Optional[genai.Client]): Runtime field used by the TTS workflow.
		audio_path (Optional[str]): Runtime field used by the TTS workflow.
		response_format (Optional[str]): Runtime field used by the TTS workflow.
		input_text (Optional[str]): Runtime field used by the TTS workflow.
		audio_bytes (Optional[bytes]): Runtime field used by the TTS workflow.
	"""
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	voice_config: Optional[ VoiceConfig ]
	speech_config: Optional[ SpeechConfig ]
	client: Optional[ genai.Client ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	
	def __init__( self, model: str='gemini-2.5-flash-preview-tts' ):
		"""Initialize instance.
		
		Purpose:
			Initializes the TTS instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			model (str): model value used by this workflow.
		"""
		super( ).__init__( )
		self.number = None
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.voice_config = None
		self.speech_config = None
		self.content_config = None
		self.client = None
		self.voice = None
		self.speed = None
		self.response = None
		self.response_format = None
		self.audio_path = None
		self.input_text = None
		self.audio_bytes = None
		self.response_modalities = [ ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
			'gemini-2.5-pro-preview-tts' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
			Returns the format options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'audio/wav' ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int=24000, channels: int=1,
		sample_width: int=2 ) -> bytes:
		"""To wave bytes.
		
		Purpose:
			Performs the to wave bytes operation for the TTS wrapper. The method preserves 
			provider-
			specific handling behind the application-facing class interface.
		
		Args:
			pcm_data (bytes): pcm data value used by this workflow.
			rate (int): rate value used by this workflow.
			channels (int): channels value used by this workflow.
			sample_width (int): sample width value used by this workflow.
		
		Returns:
			Result produced by the operation.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			import io
			import wave
			throw_if( 'pcm_data', pcm_data )
			with io.BytesIO( ) as buffer:
				with wave.open( buffer, 'wb' ) as wf:
					wf.setnchannels( channels )
					wf.setsampwidth( sample_width )
					wf.setframerate( rate )
					wf.writeframes( pcm_data )
				
				return buffer.getvalue( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'to_wave_bytes( self, **kwargs) -> bytes'
			Logger( ).write( exception )
			raise exception
	
	def normalize_voice( self, voice: Optional[ str ] = None ) -> str:
		"""Normalize voice.
		
		Purpose:
			Normalizes input values for the TTS workflow before they are passed to provider 
			calls or
			downstream processing. The method converts UI or caller-supplied values into a stable
			shape expected by the wrapper.
		
		Args:
			voice (Optional[str]): voice value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.voice_name = str( voice or '' ).strip( )
			self.valid_voices = set( self.voice_options or [ ] )
			if self.voice_name in self.valid_voices:
				return self.voice_name
			
			return 'Kore'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_voice( self, voice: Optional[str]=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_tts_prompt( self, text: str, speed: Optional[ float ] = None,
		instruct: Optional[ str ] = None ) -> str:
		"""Normalize tts prompt.
		
		Purpose:
			Normalizes input values for the TTS workflow before they are passed to provider 
			calls or
			downstream processing. The method converts UI or caller-supplied values into a stable
			shape expected by the wrapper.
		
		Args:
			text (str): text value used by this workflow.
			speed (Optional[float]): speed value used by this workflow.
			instruct (Optional[str]): instruct value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'text', text )
			self.prompt_parts = [ ]
			if instruct is not None and str( instruct ).strip( ):
				self.prompt_parts.append( str( instruct ).strip( ) )
			
			if speed is not None:
				self.speed_value = float( speed )
				if self.speed_value < 0.85:
					self.prompt_parts.append( 'Read the following text at a slow, clear pace.' )
				elif self.speed_value > 1.15:
					self.prompt_parts.append( 'Read the following text at a faster, energetic pace.' )
			
			self.prompt_parts.append( str( text ).strip( ) )
			return '\n\n'.join( self.prompt_parts )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_tts_prompt( self, **kwargs ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, filepath: str=None,
		model: str='gemini-3.1-flash-tts-preview', format: str='audio/wav', speed: float=None,
		voice: str=None, frequency: float=None, presense: float=None, max_tokens: int=None,
		instruct: str=None, temperature: float=None,
		top_p: float=None ) -> bytes | str | None:
		"""Create speech.
		
		Purpose:
			Executes the create speech workflow for the TTS wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			text (str): text value used by this workflow.
			filepath (str): filepath value used by this workflow.
			model (str): model value used by this workflow.
			format (str): format value used by this workflow.
			speed (float): speed value used by this workflow.
			voice (str): voice value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presense (float): presense value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			instruct (str): instruct value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = self.normalize_tts_prompt( text=text, speed=speed, 
				instruct=instruct )
			self.audio_path = filepath
			self.response_format = str( format or 'audio/wav' ).strip( )
			self.speed = speed
			self.voice = self.normalize_voice( voice )
			self.frequency_penalty = frequency
			self.presence_penalty = presense
			self.max_tokens = max_tokens
			self.model = str( model or self.model or 'gemini-3.1-flash-tts-preview' ).strip( )
			self.temperature = temperature
			self.top_p = top_p
			self.response_modalities = [ 'AUDIO' ]
			if self.response_format != 'audio/wav':
				raise ValueError( 'Gemini TTS wrapper currently supports local WAV output only.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini TTS model: {self.model}' )
			
			self.voice_config = VoiceConfig(
				prebuilt_voice_config=types.PrebuiltVoiceConfig( voice_name=self.voice ) )
			self.speech_config = SpeechConfig( voice_config=self.voice_config )
			self.config_kwargs = { 'response_modalities': self.response_modalities,
				'speech_config': self.speech_config }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.input_text, config=self.content_config )
			
			self.audio_bytes = None
			for part in self.response.candidates[ 0 ].content.parts:
				if getattr( part, 'inline_data', None ) is not None and part.inline_data.data:
					self.audio_bytes = self.to_wave_bytes( part.inline_data.data )
					break
			
			if self.audio_bytes is None:
				raise ValueError( 'No audio bytes were returned by Gemini TTS.' )
			
			if self.audio_path is not None and str( self.audio_path ).strip( ):
				with open( self.audio_path, 'wb' ) as f:
					f.write( self.audio_bytes )
				
				return self.audio_path
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('create_speech( self, text: str, filepath: str=None, '
			                    'model: str="gemini-3.1-flash-tts-preview", format: '
			                    'str="audio/wav", '
			                    'speed: float=None, voice: str=None, frequency: float=None, '
			                    'presense: float=None, max_tokens: int=None, instruct: str=None, '
			                    'temperature: float=None, top_p: float=None ) -> bytes | str | '
			                    'None')
			Logger( ).write( exception )
			raise exception

class Transcription( Gemini ):
	"""Transcription class.
	
	Purpose:
		Transcribes local audio files into text using Gemini audio-understanding models. The
		class normalizes audio MIME types, builds transcription prompts with optional language
		and time-window hints, uploads the audio file, and returns the generated transcript
		text.
	
	Attributes:
		client (Optional[genai.Client]): Runtime field used by the Transcription workflow.
		transcript (Optional[str]): Runtime field used by the Transcription workflow.
		file_path (Optional[str]): Runtime field used by the Transcription workflow.
		response (Optional[GenerateContentResponse]): Runtime field used by the Transcription 
		workflow.
	"""
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int=1, model: str='gemini-3-flash-preview', temperature: float=0.8,
		top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
		instruct: str=None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Transcription instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			n (int): n value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			instruct (str): instruct value used by this workflow.
		"""
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.transcript = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-3-flash-preview', 'gemini-2.0-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
			Returns the language options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'Auto', 'English', 'Spanish', 'French', 'Japanese', 'German', 'Chinese' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
			Returns the format options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'audio/wav', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg', 'audio/flac' ]
	
	def normalize_mime_type( self, path: str, mime_type: str=None ) -> str:
		"""Normalize mime type.
		
		Purpose:
			Normalizes input values for the Transcription workflow before they are passed to
			provider calls or downstream processing. The method converts UI or caller-supplied
			values into a stable shape expected by the wrapper.
		
		Args:
			path (str): path value used by this workflow.
			mime_type (str): mime type value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			import mimetypes
			
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = { 'audio/mpeg': 'audio/mp3', 'audio/x-mp3': 'audio/mp3',
				'audio/x-wav': 'audio/wav', 'audio/wave': 'audio/wav', 'audio/x-m4a': 'audio/aac',
				'audio/m4a': 'audio/aac', 'audio/mp4': 'audio/aac', 'audio/x-aiff': 'audio/aiff',
				'audio/aif': 'audio/aiff', 'audio/x-flac': 'audio/flac' }
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = { '.wav': 'audio/wav', '.mp3': 'audio/mp3', '.aiff': 'audio/aiff',
				'.aif': 'audio/aiff', '.aac': 'audio/aac', '.m4a': 'audio/aac', '.ogg': 
					'audio/ogg',
				'.flac': 'audio/flac' }
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_prompt( self, language: str=None, start_time: float=None,
		end_time: float=None ) -> str:
		"""Build prompt.
		
		Purpose:
			Builds the request component used by the Transcription workflow. The method translates
			caller options and current object state into provider-compatible configuration or
			content values.
		
		Args:
			language (str): language value used by this workflow.
			start_time (float): start time value used by this workflow.
			end_time (float): end time value used by this workflow.
		
		Returns:
			Provider-compatible request component or configuration value.
		"""
		self.prompt_parts = [ 'Generate a verbatim transcript of the speech.' ]
		
		if (language is not None and str( language ).strip( ) and str( language ).strip( ) != 
				'Auto'):
			self.prompt_parts.append(
				f'The expected spoken language is {str( language ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only transcribe the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the transcript text.' )
		return ' '.join( self.prompt_parts )
	
	def transcribe( self, path: str, model: str='gemini-3-flash-preview', language: str=None,
		mime_type: str=None, temperature: float=None, top_p: float=None,
		frequency: float=None, presence: float=None, max_tokens: int=None,
		start_time: float=None, end_time: float=None, instruct: str=None ) -> Optional[ str ]:
		"""Transcribe.
		
		Purpose:
			Executes the transcribe workflow for the Transcription wrapper. The method validates
			required inputs, prepares provider configuration, performs the requested provider or
			storage operation, captures response state, and returns the result expected by the
			application.
		
		Args:
			path (str): path value used by this workflow.
			model (str): model value used by this workflow.
			language (str): language value used by this workflow.
			mime_type (str): mime type value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			start_time (float): start time value used by this workflow.
			end_time (float): end time value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			import mimetypes
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = self.normalize_mime_type( path=self.file_path, mime_type=mime_type )
			self.prompt = self.build_prompt( language=language, start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.uploaded_file ], config=self.content_config )
			self.transcript = self.response.text
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path, model, language ) -> str'
			Logger( ).write( ex )
			raise ex

class Translation( Gemini ):
	"""Translation class.
	
	Purpose:
		Translates spoken audio into target-language text using Gemini audio-understanding
		models. The class normalizes MIME types, builds translation prompts with source and
		target language hints, uploads the audio file, and returns translated text.
	
	Attributes:
		client (Optional[genai.Client]): Runtime field used by the Translation workflow.
		target_language (Optional[str]): Runtime field used by the Translation workflow.
		source_language (Optional[str]): Runtime field used by the Translation workflow.
		file_path (Optional[str]): Runtime field used by the Translation workflow.
		response (Optional[GenerateContentResponse]): Runtime field used by the Translation 
		workflow.
	"""
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int=1, model: str='gemini-3-flash-preview', temperature: float=0.8,
		top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
		instruct: str=None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Translation instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			n (int): n value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			instruct (str): instruct value used by this workflow.
		"""
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-3-flash-preview', 'gemini-2.0-flash' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
			Returns the format options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'audio/wav', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg', 'audio/flac' ]
	
	def normalize_mime_type( self, path: str, mime_type: str=None ) -> str:
		"""Normalize mime type.
		
		Purpose:
			Normalizes input values for the Translation workflow before they are passed to provider
			calls or downstream processing. The method converts UI or caller-supplied values into a
			stable shape expected by the wrapper.
		
		Args:
			path (str): path value used by this workflow.
			mime_type (str): mime type value used by this workflow.
		
		Returns:
			Normalized value suitable for provider calls or downstream processing.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			import mimetypes
			
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = { 'audio/mpeg': 'audio/mp3', 'audio/x-mp3': 'audio/mp3',
				'audio/x-wav': 'audio/wav', 'audio/wave': 'audio/wav', 'audio/x-m4a': 'audio/aac',
				'audio/m4a': 'audio/aac', 'audio/mp4': 'audio/aac', 'audio/x-aiff': 'audio/aiff',
				'audio/aif': 'audio/aiff', 'audio/x-flac': 'audio/flac' }
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = { '.wav': 'audio/wav', '.mp3': 'audio/mp3', '.aiff': 'audio/aiff',
				'.aif': 'audio/aiff', '.aac': 'audio/aac', '.m4a': 'audio/aac', '.ogg': 
					'audio/ogg',
				'.flac': 'audio/flac' }
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
			Returns the language options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'English', 'Spanish', 'French', 'Japanese', 'German', 'Chinese' ]
	
	def build_prompt( self, target: str, source: str='Auto', start_time: float=None,
		end_time: float=None ) -> str:
		"""Build prompt.
		
		Purpose:
			Builds the request component used by the Translation workflow. The method translates
			caller options and current object state into provider-compatible configuration or
			content values.
		
		Args:
			target (str): target value used by this workflow.
			source (str): source value used by this workflow.
			start_time (float): start time value used by this workflow.
			end_time (float): end time value used by this workflow.
		
		Returns:
			Provider-compatible request component or configuration value.
		"""
		self.prompt_parts = [ f'Translate the spoken audio into {target}.' ]
		if source is not None and str( source ).strip( ) and str( source ).strip( ) != 'Auto':
			self.prompt_parts.append( f'The expected source language is '
			                          f'{str( source ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only translate the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the translated text.' )
		return ' '.join( self.prompt_parts )
	
	def translate( self, path: str, model: str='gemini-3-flash-preview',
		language: str='English', source: str='Auto', mime_type: str=None,
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, start_time: float=None,
		end_time: float=None, instruct: str=None ) -> Optional[ str ]:
		"""Translate.
		
		Purpose:
			Executes the translate workflow for the Translation wrapper. The method validates
			required inputs, prepares provider configuration, performs the requested provider or
			storage operation, captures response state, and returns the result expected by the
			application.
		
		Args:
			path (str): path value used by this workflow.
			model (str): model value used by this workflow.
			language (str): language value used by this workflow.
			source (str): source value used by this workflow.
			mime_type (str): mime type value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			start_time (float): start time value used by this workflow.
			end_time (float): end time value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.target_language = str( language or 'English' ).strip( )
			self.source_language = str( source or 'Auto' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = self.normalize_mime_type( path=self.file_path, mime_type=mime_type )
			self.prompt = self.build_prompt( target=self.target_language,
				source=self.source_language, start_time=start_time, end_time=end_time )
			
			self.config_kwargs = { }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.uploaded_file ], config=self.content_config )
			return self.response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path, model, language, source ) -> str'
			Logger( ).write( ex )
			raise ex

class Files( Gemini ):
	"""Files class.
	
	Purpose:
		Encapsulates Gemini file and document workflows used by the application. The class
		uploads local files, lists configured storage assets, retrieves remote metadata,
		summarizes documents, searches uploaded content, and deletes remote files when
		requested.
	
	Attributes:
		api_version (Optional[str]): Runtime field used by the Files workflow.
		google_api_key (Optional[str]): Runtime field used by the Files workflow.
		storage_client (Optional[storage.Client]): Runtime field used by the Files workflow.
		project_id (Optional[str]): Runtime field used by the Files workflow.
		project_location (Optional[str]): Runtime field used by the Files workflow.
		file_id (Optional[str]): Runtime field used by the Files workflow.
		bucket_id (Optional[str]): Runtime field used by the Files workflow.
		display_name (Optional[str]): Runtime field used by the Files workflow.
		mime_type (Optional[str]): Runtime field used by the Files workflow.
		file_path (Optional[str]): Runtime field used by the Files workflow.
		file_list (Optional[List[File]]): Runtime field used by the Files workflow.
		file_paths (Optional[List[str]]): Runtime field used by the Files workflow.
		file_lists (Optional[List[File]]): Runtime field used by the Files workflow.
		response (Optional[Any]): Runtime field used by the Files workflow.
		use_vertex (Optional[bool]): Runtime field used by the Files workflow.
		collections (Optional[Dict[str, str]]): Runtime field used by the Files workflow.
		documents (Optional[Dict[str, str]]): Runtime field used by the Files workflow.
	"""
	api_version: Optional[ str ]
	google_api_key: Optional[ str ]
	storage_client: Optional[ storage.Client ]
	project_id: Optional[ str ]
	project_location: Optional[ str ]
	file_id: Optional[ str ]
	bucket_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: Optional[ List[ File ] ]
	file_paths: Optional[ List[ str ] ]
	file_lists: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str='gemini-2.0-flash' ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Files instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		
		Args:
			model (str): model value used by this workflow.
		"""
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.tool_choice = None
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.files = [ ]
		self.http_options = { }
		self.storage_client = None
		self.bucket_id = None
		self.file_id = None
		self.display_name = None
		self.media_resolution = None
		self.mime_type = None
		self.file_path = None
		self.file_list = [ ]
		self.response = None
		self.collections = { }
		self.documents = { }
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""File options.
		
		Purpose:
			Returns the file options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-3.5-flash', 'gemini-3.5 flash-lite', 'gemini-3.0-flash',
			'gemini-3.0-flash-lite' ]
	
	@property
	def media_options( self ):
		"""Media options.
		
		Purpose:
			Returns the media options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
			Returns the include options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
			Returns the reasoning options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
			Returns the choice options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'AUTO', 'ANY', 'NONE', 'VALIDATED' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
			Returns the tool options exposed by this provider wrapper. This property keeps UI 
			option
			rendering centralized and gives documentation a stable location for describing 
			supported
			choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'google_search', 'google_maps', 'file_search', 'url_context', 'code_execution',
			'computer_use' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
			Returns the modality options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.
		
		Purpose:
			Returns the media options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	def upload( self, filepath: str, name: str=None ) -> File | None:
		"""Upload.
		
		Purpose:
			Executes the upload workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			filepath (str): filepath value used by this workflow.
			name (str): name value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'name', name )
			self.file_path = filepath;
			self.display_name = name
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.files.upload( path=self.file_path,
				config={ 'display_name': self.display_name } )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def list( self, model: str='gemini-3.0-flash', top_p: float=0.8, top_k: int=50,
		temperature: float=0.5, frequency: float=0.0, presence: float=0.0,
		max_tokens: int=8192, tool_choice: str='auto', stops: List[ str ]=None,
		tools: List[ str ]=None, domains: List[ str ]=None, modalities: List[ str ]=None,
		media_resolution: str='media_resolution_medium' ) -> Any | None:
		"""List.
		
		Purpose:
			Executes the list workflow for the Files wrapper. The method validates required inputs,
			prepares provider configuration, performs the requested provider or storage operation,
			captures response state, and returns the result expected by the application.
		
		Args:
			model (str): model value used by this workflow.
			top_p (float): top p value used by this workflow.
			top_k (int): top k value used by this workflow.
			temperature (float): temperature value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			tool_choice (str): tool choice value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			tools (List[str]): tools value used by this workflow.
			domains (List[str]): domains value used by this workflow.
			modalities (List[str]): modalities value used by this workflow.
			media_resolution (str): media resolution value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.files = [ ]
			self.model = model
			self.top_p = top_p
			self.top_k = top_k
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.tool_choice = tool_choice
			self.stops = stops if stops is not None else [ ]
			self.tools = tools if tools is not None else [ ]
			self.domains = domains if domains is not None else [ ]
			self.response_modalities = modalities if modalities is not None else [ ]
			self.media_resolution = media_resolution
			self.storage_client = storage.Client( )
			name = 'jeni-financial'
			prefix = 'regulations'
			bucket = self.storage_client.bucket( bucket_name=name )
			for blob in bucket.list_blobs( prefix=prefix ):
				self.files.append( blob.name )
			
			self.file_list = self.files
			return self.files
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list( self ) -> Any | None'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, file_id: str ) -> Optional[ File ]:
		"""Retrieve.
		
		Purpose:
			Executes the retrieve workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			file_id (str): file id value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str ) -> Optional[ File ]'
			Logger( ).write( ex )
			raise ex
	
	def summarize( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Summarize.
		
		Purpose:
			Executes the summarize workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			filepath (str): filepath value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'summarize_document( self, prompt, filepath, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def search( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Search.
		
		Purpose:
			Executes the search workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			filepath (str): filepath value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'search( self, prompt, filepath, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str='gemini-2.0-flash',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ]=None ) -> str | None:
		"""Survey.
		
		Purpose:
			Executes the survey workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			filepaths (List[str]): filepaths value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepaths', filepaths )
			self.prompt = prompt
			self.file_paths = filepaths
			self.model = model
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_paths )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'survey( self, prompt, filepaths, model ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def web_search( self, prompt: str, model: str='gemini-2.5-flash-lite',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Web search.
		
		Purpose:
			Executes the web search workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [ types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def search_maps( self, prompt: str, model: str='gemini-2.5-flash-lite',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Search maps.
		
		Purpose:
			Executes the search maps workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
				types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, file_id: str ) -> bool | None:
		"""Delete.
		
		Purpose:
			Executes the delete workflow for the Files wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			file_id (str): file id value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.client.files.delete( name=self.file_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'FileStore'
			ex.method = 'delete( self, file_id: str ) -> bool'
			Logger( ).write( ex )
			raise ex

class FileSearch( Gemini ):
	"""FileSearch class.
	
	Purpose:
		Manages Gemini File Search Store resources for retrieval-augmented workflows. The class
		creates, retrieves, lists, deletes, and refreshes file-search store mappings so the UI
		can expose stable display names for provider resource names.
	
	Attributes:
		client (Optional[genai.Client]): Runtime field used by the FileSearch workflow.
		response (Optional[Any]): Runtime field used by the FileSearch workflow.
		store_id (Optional[str]): Runtime field used by the FileSearch workflow.
		store_name (Optional[str]): Runtime field used by the FileSearch workflow.
		collections (Optional[Dict[str, str]]): Runtime field used by the FileSearch workflow.
		stores (Optional[List[FileSearchStore]]): Runtime field used by the FileSearch workflow.
	"""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	store_id: Optional[ str ]
	store_name: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	stores: Optional[ List[ FileSearchStore ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
			Initializes the FileSearch instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		"""
		super( ).__init__( )
		self.client = None
		self.response = None
		self.store_id = None
		self.store_name = None
		self.collections = { }
		self.stores = [ ]
		self.refresh_collections( )
	
	def refresh_collections( self ) -> Dict[ str, str ]:
		"""Refresh collections.
		
		Purpose:
			Performs the refresh collections operation for the FileSearch wrapper. The method
			preserves provider-specific handling behind the application-facing class interface.
		
		Returns:
			Result produced by the operation.
		"""
		try:
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.collections = { }
			self.stores = [ ]
			for store in self.client.file_search_stores.list( ):
				self.stores.append( store )
				self.display_name = getattr( store, 'display_name', None )
				self.resource_name = getattr( store, 'name', None )
				
				if self.resource_name is None:
					continue
				
				self.label = str( self.display_name ).strip( ) if self.display_name else str(
					self.resource_name ).strip( )
				self.collections[ self.label ] = str( self.resource_name ).strip( )
			
			return self.collections
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'refresh_collections( self ) -> Dict[ str, str ]'
			Logger( ).write( exception )
			self.collections = { }
			self.stores = [ ]
			return self.collections
	
	def create( self, name: str ) -> FileSearchStore | Any:
		"""Create.
		
		Purpose:
			Executes the create workflow for the FileSearch wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			name (str): name value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'name', name )
			self.store_name = str( name ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.create(
				config={ 'display_name': self.store_name } )
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create( self, name: str ) -> FileSearchStore | Any'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore | Any:
		"""Retrieve.
		
		Purpose:
			Executes the retrieve workflow for the FileSearch wrapper. The method validates 
			required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			store_id (str): store id value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.get( name=self.store_id )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'retrieve( self, store_id: str ) -> FileSearchStore | Any'
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ FileSearchStore ] | Any:
		"""List.
		
		Purpose:
			Executes the list workflow for the FileSearch wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			self.refresh_collections( )
			return self.stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'list( self ) -> List[ FileSearchStore ] | Any'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, force: bool=True ) -> bool | Any:
		"""Delete.
		
		Purpose:
			Executes the delete workflow for the FileSearch wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			store_id (str): store id value used by this workflow.
			force (bool): force value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.client.file_search_stores.delete( name=self.store_id,
				config={ 'force': bool( force ) } )
			self.refresh_collections( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'delete( self, store_id: str, force: bool=True ) -> bool | Any'
			Logger( ).write( exception )
			raise exception

class CloudBuckets( Gemini ):
	"""CloudBuckets class.
	
	Purpose:
		Wraps Google Cloud Storage bucket operations used as a document and asset backend. The
		class uploads local files, retrieves blob metadata, lists bucket contents, deletes
		objects, and exposes predefined collection mappings used by the application.
	
	Attributes:
		project_id (Optional[str]): Runtime field used by the CloudBuckets workflow.
		bucket_name (Optional[str]): Runtime field used by the CloudBuckets workflow.
		object_name (Optional[str]): Runtime field used by the CloudBuckets workflow.
		file_path (Optional[str]): Runtime field used by the CloudBuckets workflow.
		file_ids (Optional[List[str]]): Runtime field used by the CloudBuckets workflow.
		store_ids (Optional[List[str]]): Runtime field used by the CloudBuckets workflow.
		client (Optional[storage.Client]): Runtime field used by the CloudBuckets workflow.
		bucket (Optional[storage.Bucket]): Runtime field used by the CloudBuckets workflow.
		response (Optional[Any]): Runtime field used by the CloudBuckets workflow.
		collections (Optional[Dict[str, str]]): Runtime field used by the CloudBuckets workflow.
		documents (Optional[Dict[str, str]]): Runtime field used by the CloudBuckets workflow.
	"""
	project_id: Optional[ str ]
	bucket_name: Optional[ str ]
	object_name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	client: Optional[ storage.Client ]
	bucket: Optional[ storage.Bucket ]
	response: Optional[ Any ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
			Initializes the CloudBuckets instance with default configuration, runtime state, and
			compatibility fields required by later method calls. The constructor prepares provider
			settings and placeholders without performing request work beyond local state 
			assignment.
		"""
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.client = storage.Client( project=self.project_id )
		self.bucket_name = None
		self.object_name = None
		self.file_path = None
		self.media_resolution = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.bucket = None
		self.response = None
		self.collections = { 'Federal Financial Data': 'jeni-financial/data',
			'Federal Financial Regulations': 'jeni-financial/regulations',
			'DoW Financial Data': 'jeni-dow/budget/data',
			'DoW Financial Regulations': 'jeni-dow/budget/regulations',
			'DoA Financial Data': 'jenni-doa/Financial Data', }
		self.documents = { 'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
			'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
			'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
			'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn' }
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
			Returns the model options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		Returns:
			Available option values or configured wrapper values.
		"""
		return [ 'gemini-2.5-flash', 'gemini-2.5 flash image', 'gemini-2.5 flash-tts',
			'gemini-2.5 flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite' ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.
		
		Purpose:
			Returns the media options exposed by this provider wrapper. This property keeps UI
			option rendering centralized and gives documentation a stable location for describing
			supported choices.
		
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	def create( self, bucket: str, name: str ) -> bool | None:
		"""Create.
		
		Purpose:
			Executes the create workflow for the CloudBuckets wrapper. The method validates 
			required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			bucket (str): bucket value used by this workflow.
			name (str): name value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def upload( self, path: str, bucket: str, name: str=None ) -> Blob | None:
		"""Upload.
		
		Purpose:
			Executes the upload workflow for the CloudBuckets wrapper. The method validates 
			required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			path (str): path value used by this workflow.
			bucket (str): bucket value used by this workflow.
			name (str): name value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'bucket', bucket )
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = name or path.split( '/' )[ -1 ]
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.upload_from_filename( self.file_path )
			self.response = blob
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'upload( self, path, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def retrieve( self, bucket: str, name: str ) -> Blob | None:
		"""Retrieve.
		
		Purpose:
			Executes the retrieve workflow for the CloudBuckets wrapper. The method validates
			required inputs, prepares provider configuration, performs the requested provider or
			storage operation, captures response state, and returns the result expected by the
			application.
		
		Args:
			bucket (str): bucket value used by this workflow.
			name (str): name value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.get_blob( self.object_name )
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
	
	def list( self, bucket: str ):
		"""List.
		
		Purpose:
			Executes the list workflow for the CloudBuckets wrapper. The method validates required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			bucket (str): bucket value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.bucket = self.client.bucket( self.bucket_name )
			blobs = list( self.bucket.list_blobs( ) )
			self.documents = { blob.name: blob.id for blob in blobs }
			return blobs
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'list( self, bucket )'
			Logger( ).write( ex )
			raise ex
	
	def web_search( self, prompt: str, model: str='gemini-2.5-flash-lite',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Web search.
		
		Purpose:
			Executes the web search workflow for the CloudBuckets wrapper. The method validates
			required inputs, prepares provider configuration, performs the requested provider or
			storage operation, captures response state, and returns the result expected by the
			application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
				types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def search_maps( self, prompt: str, model: str='gemini-2.5-flash-lite',
		temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float=None, max_tokens: int=None, stops: List[ str ] = None,
		instruct: str=None ) -> str | None:
		"""Search maps.
		
		Purpose:
			Executes the search maps workflow for the CloudBuckets wrapper. The method validates
			required inputs, prepares provider configuration, performs the requested provider or
			storage operation, captures response state, and returns the result expected by the
			application.
		
		Args:
			prompt (str): prompt value used by this workflow.
			model (str): model value used by this workflow.
			temperature (float): temperature value used by this workflow.
			top_p (float): top p value used by this workflow.
			frequency (float): frequency value used by this workflow.
			presence (float): presence value used by this workflow.
			max_tokens (int): max tokens value used by this workflow.
			stops (List[str]): stops value used by this workflow.
			instruct (str): instruct value used by this workflow.
		
		Returns:
			Result produced by the requested provider, file, audio, image, or storage workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.instructions = instruct
			self.tool_config = [
				types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, bucket: str, name: str ):
		"""Delete.
		
		Purpose:
			Executes the delete workflow for the CloudBuckets wrapper. The method validates 
			required
			inputs, prepares provider configuration, performs the requested provider or storage
			operation, captures response state, and returns the result expected by the application.
		
		Args:
			bucket (str): bucket value used by this workflow.
			name (str): name value used by this workflow.
		
		Raises:
			Error: Re-raised after provider, validation, or storage exceptions are wrapped and 
			logged.
			ValueError: Raised when local validation detects an invalid required value.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			Logger( ).write( ex )
			raise ex
