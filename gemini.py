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
import io
import json
import os
import requests
import PIL.Image
from typing import Any, Callable, Dict, List, Optional, Union, Set, Tuple
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

class Gemini( ):
	"""Shared configuration state for Gemini provider wrappers.

	Purpose:
		Initializes the API credentials, model parameters, generation settings, response
		configuration, and tool selections shared by the specialized Gemini wrapper classes.
		The base class performs no provider requests and contains no workflow-specific processing.

	Attributes:
		google_api_key (Optional[str]): Google API key loaded from application configuration.
		gemini_api_key (Optional[str]): Gemini API key loaded from application configuration.
		model (Optional[str]): Active Gemini model identifier.
		api_version (Optional[str]): API version selected by a specialized wrapper.
		temperature (Optional[float]): Sampling temperature.
		top_p (Optional[float]): Top-p sampling value.
		top_k (Optional[int]): Top-k sampling value.
		candidate_count (Optional[int]): Requested candidate count.
		frequency_penalty (Optional[float]): Frequency-penalty value.
		presence_penalty (Optional[float]): Presence-penalty value.
		max_tokens (Optional[int]): Maximum output-token count.
		instructions (Optional[str]): System instruction text.
		prompt (Optional[str]): Active user prompt.
		response_format (Optional[str]): Requested response format.
		number (Optional[int]): Requested number of generated results.
		response_modalities (List[str]): Requested response modalities.
		stops (List[str]): Stop sequences.
		domains (List[str]): Domain restrictions used by supported search workflows.
		tools (List[str]): Selected provider tools.
		tool_choice (Optional[str]): Tool-selection behavior.
		content_response (Optional[GenerateContentResponse]): Latest content response.
		image_response (Optional[GenerateImagesResponse]): Latest image-generation response.
		content_config (Optional[GenerateContentConfig]): Content-generation configuration.
		function_config (Optional[FunctionCallingConfig]): Function-calling configuration.
		thought_config (Optional[ThinkingConfig]): Thinking configuration.
		genimg_config (Optional[GenerateImagesConfig]): Image-generation configuration.
		image_config (Optional[ImageConfig]): Image-output configuration.
		tool_config (Optional[List[types.Tool]]): Provider tool configuration.
	"""

	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_tokens: Optional[ int ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	response_modalities: List[ str ]
	stops: List[ str ]
	domains: List[ str ]
	tools: List[ str ]
	tool_choice: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]

	def __init__( self ) -> None:
		"""Initialize shared Gemini wrapper state.

		Purpose:
			Initializes common credentials, model parameters, request configuration, tool
			selections, and response placeholders used by the specialized Gemini wrappers. The
			constructor performs local state assignment only.

		Returns:
			None: This method initializes object state through side effects.
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
		self.domains = [ ]
		self.tools = [ ]
		self.tool_choice = None
		self.content_response = None
		self.image_response = None
		self.content_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.image_config = None
		self.tool_config = None

class Chat( Gemini ):
	"""Gemini text-generation wrapper.

	Purpose:
		Executes text-generation, grounding, structured-output, streaming, URL Context,
		File Search, Google Search, Google Maps, and Code Execution workflows through the
		Gemini Interactions API. The class preserves the application-facing contract used by
		Jeni while maintaining conversation history through the existing client-managed
		application state.

	Attributes:
		use_vertex (bool): Whether Vertex AI configuration is enabled.
		http_options (HttpOptions): Gemini client HTTP configuration.
		client (Optional[genai.Client]): Gemini SDK client.
		storage_client (Optional[storage.Client]): Optional Google Cloud Storage client.
		contents (Optional[List[Dict[str, Any]]]): Complete Interactions input timeline.
		input_steps (List[Dict[str, Any]]): Interactions input steps submitted to Gemini.
		image_uri (Optional[str]): Optional image URI retained for interface compatibility.
		audio_uri (Optional[str]): Optional audio URI retained for interface compatibility.
		file_path (Optional[str]): Optional local file path retained for compatibility.
		files (List[str]): File identifiers retained for interface compatibility.
		content_block (str): Additional content prepended to the active prompt.
		context (List[Dict[str, Any]]): Existing application-managed conversation history.
		urls (List[str]): URL-context values.
		max_urls (int): Maximum URL values included in the prompt.
		response_schema (Optional[Dict[str, Any]]): Parsed structured-output schema.
		safety_profile (str): Safety-profile value retained for UI compatibility.
		safety_settings (Optional[List[SafetySetting]]): Optional Gemini safety settings.
		file_search_store_names (List[str]): File Search Store resource names.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		interaction_id (Optional[str]): Identifier of the most recent Interaction.
		steps (List[Any]): Steps returned by the most recent Interaction.
		response (Optional[Any]): Raw response used by application token accounting.
		output_text (str): Generated response text.
		grounding_sources (List[Dict[str, Any]]): Normalized grounding citations.
		generation_config (Dict[str, Any]): Interactions generation configuration.
		interaction_response_format (Optional[Any]): Interactions response-format value.
		tool_objects (List[Dict[str, Any]]): Interactions server-side tool definitions.
		stream (bool): Whether streaming is enabled.
		stream_handler (Optional[Callable[[str], None]]): Text-delta callback.
	"""
	
	use_vertex: bool
	http_options: HttpOptions
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ List[ Dict[ str, Any ] ] ]
	input_steps: List[ Dict[ str, Any ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: List[ str ]
	content_block: str
	context: List[ Dict[ str, Any ] ]
	urls: List[ str ]
	max_urls: int
	response_schema: Optional[ Dict[ str, Any ] ]
	safety_profile: str
	safety_settings: Optional[ List[ SafetySetting ] ]
	file_search_store_names: List[ str ]
	interaction: Optional[ Any ]
	interaction_id: Optional[ str ]
	steps: List[ Any ]
	response: Optional[ Any ]
	output_text: str
	grounding_sources: List[ Dict[ str, Any ] ]
	generation_config: Dict[ str, Any ]
	interaction_response_format: Optional[ Any ]
	tool_objects: List[ Dict[ str, Any ] ]
	stream: bool
	stream_handler: Optional[ Callable[ [ str ], None ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize the Chat wrapper.

		Purpose:
			Initializes text-generation configuration, Interactions request state, grounding
			state, conversation-history state, and response placeholders. The constructor
			performs local state assignment only.

		Args:
			model (str): Default Gemini text-generation model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.use_vertex = False
		self.client = None
		self.storage_client = None
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
		self.response_schema = None
		self.response_modalities = [ ]
		self.media_resolution = None
		self.tool_choice = None
		self.tools = [ ]
		self.tool_objects = [ ]
		self.generation_config = { }
		self.interaction_response_format = None
		self.safety_profile = ''
		self.safety_settings = None
		self.contents = None
		self.input_steps = [ ]
		self.content_block = ''
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
		self.steps = [ ]
		self.response = None
		self.content_response = None
		self.output_text = ''
		self.grounding_metadata = None
		self.grounding_sources = [ ]
		self.stream = False
		self.stream_handler = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini text-generation models.

		Purpose:
			Provides the text-generation model identifiers exposed by the Jeni Text,
			Document Q&A, File Search Stores, and Google Cloud Buckets interfaces.

		Returns:
			List[str]: Supported Gemini text-generation model identifiers.
		"""
		return [ 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro',
			'gemini-3.1-flash-lite', 'gemini-3.1-pro-preview', 'gemini-3.5-flash',
			'gemini-3.6-flash', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported Interactions server-side tools.

		Purpose:
			Provides the server-side tools exposed by the Jeni Text interface.

		Returns:
			List[str]: Supported Interactions tool identifiers.
		"""
		return [ 'google_search', 'google_maps', 'url_context', 'file_search', 'code_execution', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported thinking levels.

		Purpose:
			Provides the thinking-level values exposed by the Jeni model controls.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Purpose:
			Preserves the media-resolution option contract used by the Jeni interface.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice values.

		Purpose:
			Provides the tool-selection values accepted by the Interactions API.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'any', 'none', 'validated', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return compatibility include options.

		Purpose:
			Preserves the option values consumed by existing Jeni controls while citations
			and tool execution are retrieved from typed Interaction steps.

		Returns:
			List[str]: Existing Jeni include-option values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported response modalities.

		Purpose:
			Provides response-modality values used by the Jeni Text interface.

		Returns:
			List[str]: Supported response modalities.
		"""
		return [ '', 'text', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported text response MIME types.

		Purpose:
			Provides text and structured-output MIME types used by the Text interface.

		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Return tools supported by the selected model.

		Purpose:
			Builds the tool list consumed by the Jeni Text controls and conditionally includes
			Google Maps for models that expose that capability.

		Args:
			model (str): Gemini model identifier.

		Returns:
			List[str]: Tool identifiers supported by the selected model.

		Raises:
			Error: Raised when validation or tool-list construction fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.options = [ 'google_search', 'url_context', 'file_search', 'code_execution', ]
			
			if self.supports_google_maps( self.model ):
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
			Centralizes Google Maps feature gating for the Jeni Text controls.

		Args:
			model (str): Gemini model identifier.

		Returns:
			bool: True when Google Maps may be exposed; otherwise False.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.maps_models = { 'gemini-3.5-flash', 'gemini-3.6-flash', }
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'supports_google_maps( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_value( self, value: Any ) -> Any:
		"""Convert an SDK value into standard Python data.

		Purpose:
			Recursively converts SDK response models, dictionaries, and sequences into
			ordinary Python values used by history and citation processing.

		Args:
			value (Any): SDK or Python value to normalize.

		Returns:
			Any: Equivalent standard Python value.

		Raises:
			Error: Raised when value normalization fails.
		"""
		try:
			self.value = value
			
			if self.value is None or isinstance( self.value, (str, int, float, bool) ):
				return self.value
			
			if isinstance( self.value, dict ):
				return { key: self.normalize_value( item ) for key, item in self.value.items( ) }
			
			if isinstance( self.value, (list, tuple, set) ):
				return [ self.normalize_value( item ) for item in self.value ]
			
			if hasattr( self.value, 'model_dump' ):
				self.value_data = self.value.model_dump( exclude_none=True )
				return self.normalize_value( self.value_data )
			
			if hasattr( self.value, 'to_dict' ):
				self.value_data = self.value.to_dict( )
				return self.normalize_value( self.value_data )
			
			return str( self.value )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'normalize_value( self, value: Any ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def normalize_context( self,
		context: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Convert application history into Interactions steps.

		Purpose:
			Preserves existing Interactions steps and converts Jeni role/content dictionaries
			into ``user_input`` and ``model_output`` steps for stateless conversation history.

		Args:
			context (Optional[List[Dict[str, Any]]]): Existing conversation history.

		Returns:
			List[Dict[str, Any]]: Interactions-compatible history steps.

		Raises:
			Error: Raised when conversation history cannot be normalized.
		"""
		try:
			self.context = context if isinstance( context, list ) else [ ]
			self.history_steps: List[ Dict[ str, Any ] ] = [ ]
			
			for message in self.context:
				if not isinstance( message, dict ):
					continue
				
				self.context_type = str( message.get( 'type', '' ) or '' ).strip( )
				if self.context_type:
					self.context_step = self.normalize_value( message )
					if isinstance( self.context_step, dict ) and self.context_step:
						self.history_steps.append( self.context_step )
					
					continue
				
				self.message_role = str( message.get( 'role', '' ) or '' ).strip( ).lower( )
				self.message_content = message.get( 'content', '' )
				
				if isinstance( self.message_content, list ):
					self.message_text = '\n'.join(
						str( item ).strip( ) for item in self.message_content if
						item is not None and str( item ).strip( ) )
				else:
					self.message_text = str( self.message_content or '' ).strip( )
				
				if not self.message_text:
					continue
				
				if self.message_role in ('assistant', 'model'):
					self.step_type = 'model_output'
				else:
					self.step_type = 'user_input'
				
				self.history_steps.append( { 'type': self.step_type,
					'content': [ { 'type': 'text', 'text': self.message_text, }, ], } )
			
			self.context = self.history_steps
			return self.context
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = (
				'normalize_context( self, context: Optional[ List[ Dict[ str, Any ] ] ] ) '
				'-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: Optional[ List[ str ] ] = None, max_urls: int = 0 ) -> List[ str ]:
		"""Build the normalized URL list.

		Purpose:
			Removes blank and duplicate URL values while preserving order and applies the
			configured maximum before URL values are supplied to the active request.

		Args:
			urls (Optional[List[str]]): Candidate URL values.
			max_urls (int): Maximum URLs to retain; zero retains all values.

		Returns:
			List[str]: Normalized URL values.

		Raises:
			Error: Raised when URL normalization fails.
		"""
		try:
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.normalized_urls: List[ str ] = [ ]
			
			for url in self.urls:
				if url is None:
					continue
				
				self.url = str( url ).strip( )
				if not self.url:
					continue
				
				if self.url not in self.normalized_urls:
					self.normalized_urls.append( self.url )
			
			if self.max_urls > 0:
				self.normalized_urls = self.normalized_urls[ :self.max_urls ]
			
			self.urls = self.normalized_urls
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_urls( self, urls: Optional[ List[ str ] ], max_urls: int ) '
			                    '-> List[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, content: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, urls: Optional[ List[ str ] ] = None,
		max_urls: int = 0 ) -> List[ Dict[ str, Any ] ]:
		"""Build the complete Interactions input timeline.

		Purpose:
			Combines existing client-managed conversation history with optional content,
			reference URLs, and the current user prompt.

		Args:
			prompt (str): Current user prompt.
			content (str): Optional content prepended to the prompt.
			context (Optional[List[Dict[str, Any]]]): Existing conversation history.
			urls (Optional[List[str]]): URL-context values.
			max_urls (int): Maximum number of URLs.

		Returns:
			List[Dict[str, Any]]: Complete Interactions input timeline.

		Raises:
			Error: Raised when validation or input construction fails.
			ValueError: Raised when ``prompt`` is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.content_block = content
			self.context = context if isinstance( context, list ) else [ ]
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.input_steps = self.normalize_context( self.context )
			self.urls = self.build_urls( self.urls, self.max_urls )
			self.current_parts: List[ str ] = [ ]
			
			if self.content_block and self.content_block.strip( ):
				self.current_parts.append( self.content_block.strip( ) )
			
			if self.urls:
				self.current_parts.append(
					'Reference URLs:\n' + '\n'.join( f'- {url}' for url in self.urls ) )
			
			self.current_parts.append( self.prompt.strip( ) )
			self.current_text = '\n\n'.join( part for part in self.current_parts if part ).strip( )
			
			self.input_steps.append( { 'type': 'user_input',
				'content': [ { 'type': 'text', 'text': self.current_text, }, ], } )
			
			self.contents = self.input_steps
			return self.input_steps
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_input( self, prompt: str, content: str, '
			                    'context: Optional[ List[ Dict[ str, Any ] ] ], '
			                    'urls: Optional[ List[ str ] ], max_urls: int ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float = 0.0, top_p: float = 0.0, top_k: int
	= 0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, reasoning: str = '',
		tool_choice: Optional[ str ] = None ) -> Dict[ str, Any ]:
		"""Build the Interactions generation configuration.

		Purpose:
			Converts supported Jeni inference controls into the Interactions generation
			configuration without submitting blank or zero-valued optional controls.

		Args:
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			reasoning (str): Thinking-level value.
			tool_choice (Optional[str]): Tool-selection behavior.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when generation configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.reasoning = reasoning
			self.tool_choice = tool_choice
			self.generation_config = { }
			
			if self.temperature > 0.0:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p > 0.0:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.generation_config[ 'top_k' ] = self.top_k
			
			if self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = self.max_tokens
			
			self.stop_sequences = [ str( value ).strip( ) for value in self.stops if
				value is not None and str( value ).strip( ) ]
			
			if self.stop_sequences:
				self.generation_config[ 'stop_sequences' ] = self.stop_sequences
			
			self.thinking_level = str( self.reasoning or '' ).strip( ).lower( )
			
			if self.thinking_level not in ('', 'thinking_level_unspecified', 'unspecified',):
				self.generation_config[ 'thinking_level' ] = self.thinking_level
			
			self.tool_choice_value = str( self.tool_choice or '' ).strip( ).lower( )
			
			if self.tool_choice_value:
				self.generation_config[ 'tool_choice' ] = self.tool_choice_value
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_generation_config( self, temperature: float, top_p: float, '
			                    'top_k: int, max_tokens: int, stops: Optional[ List[ str ] ], '
			                    'reasoning: str, tool_choice: Optional[ str ] ) '
			                    '-> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any = None ) -> Optional[ Dict[ str, Any ] ]:
		"""Parse an optional structured-output schema.

		Purpose:
			Accepts a dictionary or JSON string and converts it into the JSON Schema mapping
			required by the Interactions API.

		Args:
			response_schema (Any): Optional JSON Schema dictionary or JSON string.

		Returns:
			Optional[Dict[str, Any]]: Parsed JSON Schema, or None when absent.

		Raises:
			Error: Raised when a nonblank schema cannot be parsed.
		"""
		try:
			self.response_schema = response_schema
			
			if self.response_schema is None:
				return None
			
			if isinstance( self.response_schema, dict ):
				return self.response_schema
			
			self.schema_text = str( self.response_schema ).strip( )
			if not self.schema_text:
				self.response_schema = None
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
			exception.method = ('parse_response_schema( self, response_schema: Any ) '
			                    '-> Optional[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_format: str = '', response_schema: Any = None,
		modalities: Optional[ List[ str ] ] = None ) -> Optional[ Any ]:
		"""Build the Interactions response format.

		Purpose:
			Converts the Jeni MIME type, response modalities, and optional JSON Schema into
			the polymorphic response-format structure used by the current Interactions API.

		Args:
			response_format (str): Requested text MIME type.
			response_schema (Any): Optional JSON Schema mapping or JSON string.
			modalities (Optional[List[str]]): Requested response modalities.

		Returns:
			Optional[Any]: Interactions response-format value.

		Raises:
			Error: Raised when response-format construction fails.
		"""
		try:
			self.response_format = response_format
			self.response_schema = response_schema
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.response_schema = self.parse_response_schema( self.response_schema )
			self.response_mime_type = str( self.response_format or '' ).strip( )
			self.normalized_modalities = [ str( value ).strip( ).lower( ) for value in
				self.response_modalities if value is not None and str( value ).strip( ) ]
			
			if self.normalized_modalities and 'text' not in self.normalized_modalities:
				self.interaction_response_format = None
				return None
			
			if self.response_schema is not None:
				self.interaction_response_format = [
					{ 'type': 'text', 'mime_type': 'application/json',
						'schema': self.response_schema, }, ]
				return self.interaction_response_format
			
			if self.response_mime_type in ('application/json', 'text/x.enum'):
				self.interaction_response_format = [
					{ 'type': 'text', 'mime_type': self.response_mime_type, }, ]
				return self.interaction_response_format
			
			self.interaction_response_format = None
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_response_format( self, response_format: str, '
			                    'response_schema: Any, modalities: Optional[ List[ str ] ] ) '
			                    '-> Optional[ Any ]')
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: Optional[ List[ str ] ] = None,
		urls: Optional[ List[ str ] ] = None,
		file_search_store_names: Optional[ List[ str ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build Interactions server-side tools.

		Purpose:
			Translates Jeni tool identifiers into Interactions tool declarations. URL Context
			is enabled when selected or when URLs are supplied. File Search is enabled when
			File Search Store resource names are supplied.

		Args:
			tools (Optional[List[str]]): Selected tool identifiers.
			urls (Optional[List[str]]): Normalized URL-context values.
			file_search_store_names (Optional[List[str]]): File Search Store names.

		Returns:
			List[Dict[str, Any]]: Interactions server-side tool declarations.

		Raises:
			Error: Raised when tool construction fails.
		"""
		try:
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.file_search_store_names = (
				file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
			self.selected_tools = [ str( value ).strip( ).lower( ) for value in self.tools if
				value is not None and str( value ).strip( ) ]
			self.file_search_store_names = [ str( value ).strip( ) for value in
				self.file_search_store_names if value is not None and str( value ).strip( ) ]
			self.tool_objects = [ ]
			
			if 'google_search' in self.selected_tools:
				self.tool_objects.append( { 'type': 'google_search', } )
			
			if ('google_maps' in self.selected_tools and self.supports_google_maps( self.model )):
				self.tool_objects.append( { 'type': 'google_maps', } )
			
			if 'url_context' in self.selected_tools or self.urls:
				self.tool_objects.append( { 'type': 'url_context', } )
			
			if 'code_execution' in self.selected_tools:
				self.tool_objects.append( { 'type': 'code_execution', } )
			
			if self.file_search_store_names:
				self.tool_objects.append( { 'type': 'file_search',
					'file_search_store_names': self.file_search_store_names, } )
			
			return self.tool_objects
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_tools( self, tools: Optional[ List[ str ] ], '
			                    'urls: Optional[ List[ str ] ], '
			                    'file_search_store_names: Optional[ List[ str ] ] ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def append_source( self, source: Dict[ str, Any ], default_type: str ) -> None:
		"""Append one normalized grounding source.

		Purpose:
			Converts provider citation and tool-result fields into the stable source shape
			consumed by Jeni and prevents duplicate records.

		Args:
			source (Dict[str, Any]): Provider citation or result mapping.
			default_type (str): Source type used when omitted by the mapping.

		Returns:
			None: This method updates source state through side effects.

		Raises:
			Error: Raised when validation or source normalization fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'source', source )
			throw_if( 'default_type', default_type )
			self.source = source
			self.default_type = default_type
			self.source_type = str(
				self.source.get( 'type', self.default_type ) or self.default_type ).strip( )
			self.source_title = str(
				self.source.get( 'title' ) or self.source.get( 'display_name' ) or self.source.get(
					'file_name' ) or self.source.get( 'name' ) or '' ).strip( )
			self.source_url = str(
				self.source.get( 'uri' ) or self.source.get( 'url' ) or self.source.get(
					'source_url' ) or '' ).strip( )
			self.source_text = str(
				self.source.get( 'text' ) or self.source.get( 'snippet' ) or self.source.get(
					'quote' ) or self.source.get( 'search_suggestions' ) or '' ).strip( )
			self.source_file_id = str(
				self.source.get( 'file_id' ) or self.source.get( 'file_name' ) or self.source.get(
					'document_name' ) or '' ).strip( )
			
			if not any( [ self.source_title, self.source_url, self.source_text,
				self.source_file_id, ] ):
				return
			
			self.source_key = (self.source_type, self.source_url or self.source_file_id,
				self.source_text,)
			
			if self.source_key in self.source_keys:
				return
			
			self.source_keys.add( self.source_key )
			self.source_values.append(
				{ 'type': self.source_type, 'title': self.source_title or None,
					'snippet': self.source_text or None, 'url': self.source_url or None,
					'files_id': self.source_file_id or None, 'metadata': self.source, } )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('append_source( self, source: Dict[ str, Any ], '
			                    'default_type: str ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def extract_grounding_sources( self, interaction: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract grounding citations from an Interaction.

		Purpose:
			Collects model-output annotations and supported server-side tool-result data
			from the current Interaction steps.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			List[Dict[str, Any]]: Normalized grounding-source records.

		Raises:
			Error: Raised when validation or citation extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.source_interaction = interaction
			self.source_values: List[ Dict[ str, Any ] ] = [ ]
			self.source_keys: Set[ Tuple[ str, str, str ] ] = set( )
			self.source_steps = getattr( self.source_interaction, 'steps', None ) or [ ]
			
			for step in self.source_steps:
				self.source_step_type = str( getattr( step, 'type', '' ) or '' ).strip( )
				
				if self.source_step_type == 'model_output':
					self.source_content = getattr( step, 'content', None ) or [ ]
					
					for block in self.source_content:
						self.annotations = getattr( block, 'annotations', None ) or [ ]
						
						for annotation in self.annotations:
							self.annotation_value = self.normalize_value( annotation )
							
							if not isinstance( self.annotation_value, dict ):
								continue
							
							self.append_source( self.annotation_value, str(
								self.annotation_value.get( 'type', 'citation' ) or 'citation' ) )
				
				elif self.source_step_type in ('google_search_result', 'file_search_result',
					'url_context_result', 'google_maps_result', 'code_execution_result',):
					self.result_value = self.normalize_value( getattr( step, 'result', None ) )
					
					if isinstance( self.result_value, list ):
						for result in self.result_value:
							if isinstance( result, dict ) and result:
								self.append_source( result, self.source_step_type )
					
					elif isinstance( self.result_value, dict ) and self.result_value:
						self.append_source( self.result_value, self.source_step_type )
			
			return self.source_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('extract_grounding_sources( self, interaction: Any ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def capture_interaction( self, interaction: Any ) -> None:
		"""Capture a completed Gemini Interaction.

		Purpose:
			Stores the raw response, identifier, output steps, generated text, and grounding
			sources on the members consumed by the Jeni application.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			None: This method updates response state through side effects.

		Raises:
			Error: Raised when validation or response extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.response = self.interaction
			self.content_response = self.interaction
			self.interaction_id = getattr( self.interaction, 'id', None )
			self.steps = list( getattr( self.interaction, 'steps', None ) or [ ] )
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			self.grounding_sources = self.extract_grounding_sources( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('capture_interaction( self, interaction: Any ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""Return sources from the most recent Interaction.

		Purpose:
			Provides normalized grounding-source records consumed by Jeni response renderers.

		Returns:
			List[Dict[str, Any]]: Grounding sources from the latest response.
		"""
		return list( self.grounding_sources )
	
	def get_structured_history( self ) -> List[ Dict[ str, Any ] ]:
		"""Return the complete stateless conversation history.

		Purpose:
			Combines the submitted input timeline with the output steps returned by the latest
			Interaction so Jeni can preserve client-managed conversation state.

		Returns:
			List[Dict[str, Any]]: Complete Interactions-compatible history.

		Raises:
			Error: Raised when output steps cannot be normalized.
		"""
		try:
			self.structured_history: List[ Dict[ str, Any ] ] = [ ]
			
			for step in self.input_steps:
				self.input_step = self.normalize_value( step )
				if isinstance( self.input_step, dict ) and self.input_step:
					self.structured_history.append( self.input_step )
			
			for step in self.steps:
				self.output_step = self.normalize_value( step )
				if isinstance( self.output_step, dict ) and self.output_step:
					self.structured_history.append( self.output_step )
			
			return self.structured_history
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('get_structured_history( self ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def generate_text_stream( self ) -> str:
		"""Execute the prepared streaming Interactions request.

		Purpose:
			Submits the active request, forwards text deltas to the application callback,
			accumulates the complete generated text, and captures final usage metadata.

		Returns:
			str: Complete generated text.

		Raises:
			Error: Raised when streaming execution or event handling fails.
		"""
		try:
			self.request = { 'model': self.model, 'input': self.input_steps, 'stream': True,
				'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			if self.generation_config:
				self.request[ 'generation_config' ] = self.generation_config
			
			if self.interaction_response_format is not None:
				self.request[ 'response_format' ] = (self.interaction_response_format)
			
			self.stream_response = self.client.interactions.create( **self.request )
			self.text_chunks: List[ str ] = [ ]
			self.completed_interaction = None
			
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
					self.completed_interaction = getattr( event, 'interaction', None )
				
				elif self.event_type == 'error':
					self.stream_error = getattr( event, 'error', None )
					self.stream_message = str( getattr( self.stream_error, 'message',
						'' ) or self.stream_error or 'Gemini streaming request failed.' )
					raise RuntimeError( self.stream_message )
			
			self.output_text = ''.join( self.text_chunks ).strip( )
			self.steps = [ ]
			
			if self.output_text:
				self.steps.append( { 'type': 'model_output',
					'content': [ { 'type': 'text', 'text': self.output_text, }, ], } )
			
			if self.completed_interaction is not None:
				self.interaction = self.completed_interaction
				self.response = self.completed_interaction
				self.content_response = self.completed_interaction
				self.interaction_id = getattr( self.completed_interaction, 'id', None )
			else:
				self.response = self.stream_response
				self.content_response = self.stream_response
			
			self.grounding_sources = [ ]
			
			if not self.output_text:
				raise ValueError( 'Gemini returned an empty streaming response.' )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text_stream( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, number: int = 1, temperature: float = 0.0,
		top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, instruct: str = '',
		response_format: str = '', tools: Optional[ List[ str ] ] = None,
		tool_choice: Optional[ str ] = None, reasoning: str = '',
		modalities: Optional[ List[ str ] ] = None, media_resolution: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, content: str = '',
		urls: Optional[ List[ str ] ] = None, max_urls: int = 0, response_schema: Any = '',
		safety_profile: str = '', file_search_store_names: Optional[ List[ str ] ] = None,
		stream: bool = False,
		stream_handler: Optional[ Callable[ [ str ], None ] ] = None ) -> str:
		"""Generate text through the Gemini Interactions API.

		Purpose:
			Preserves Jeni's text-generation method contract while routing model execution
			through the Interactions API. Existing client-managed history is converted into
			Interactions steps and returned through ``get_structured_history()`` after the call.

		Args:
			prompt (str): Current user prompt.
			model (str): Gemini model identifier.
			number (int): Candidate-count value retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			instruct (str): System instruction text.
			response_format (str): Requested response MIME type.
			tools (Optional[List[str]]): Server-side tool identifiers.
			tool_choice (Optional[str]): Tool-selection behavior.
			reasoning (str): Thinking-level value.
			modalities (Optional[List[str]]): Requested output modalities.
			media_resolution (str): Media-resolution value retained for compatibility.
			context (Optional[List[Dict[str, Any]]]): Client-managed conversation history.
			content (str): Additional application content.
			urls (Optional[List[str]]): URL-context values.
			max_urls (int): Maximum number of URLs.
			response_schema (Any): Optional JSON Schema mapping or JSON string.
			safety_profile (str): Safety-profile value retained for compatibility.
			file_search_store_names (Optional[List[str]]): File Search Store names.
			stream (bool): Whether to stream response text.
			stream_handler (Optional[Callable[[str], None]]): Text-delta callback.

		Returns:
			str: Generated response text.

		Raises:
			Error: Raised when validation, request construction, or execution fails.
			ValueError: Raised when a required prompt, model, or API key is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.candidate_count = self.number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = instruct
			self.response_format = response_format
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.tool_choice = tool_choice
			self.reasoning = reasoning
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.media_resolution = media_resolution
			self.context = context if isinstance( context, list ) else [ ]
			self.content_block = content
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.response_schema = response_schema
			self.safety_profile = safety_profile
			self.file_search_store_names = (
				file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
			self.stream = stream
			self.stream_handler = stream_handler
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			
			self.input_steps = self.build_input( prompt=self.prompt, content=self.content_block,
				context=self.context, urls=self.urls, max_urls=self.max_urls )
			
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, top_k=self.top_k, max_tokens=self.max_tokens, stops=self.stops,
				reasoning=self.reasoning, tool_choice=self.tool_choice )
			
			self.interaction_response_format = self.build_response_format(
				response_format=self.response_format, response_schema=self.response_schema,
				modalities=self.response_modalities )
			
			self.tool_objects = self.build_tools( tools=self.tools, urls=self.urls,
				file_search_store_names=self.file_search_store_names )
			
			if self.stream:
				return self.generate_text_stream( )
			
			self.request = { 'model': self.model, 'input': self.input_steps, 'stream': False,
				'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			if self.generation_config:
				self.request[ 'generation_config' ] = self.generation_config
			
			if self.interaction_response_format is not None:
				self.request[ 'response_format' ] = (self.interaction_response_format)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.capture_interaction( self.interaction )
			
			if not self.output_text:
				raise ValueError( 'Gemini returned an empty Interactions response.' )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, **kwargs ) -> str'
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Gemini image-generation, analysis, and editing wrapper.

	Purpose:
		Executes image generation, image understanding, and image editing through the Gemini
		Interactions API. The class converts local images into Interactions image-content
		blocks, builds image and text response formats, configures supported Google Search
		grounding, captures generated image or text output, and preserves the application-facing
		method contracts consumed by the Jeni Images mode.

	Attributes:
		client (Optional[genai.Client]): Active Gemini client.
		http_options (HttpOptions): Gemini client HTTP configuration.
		aspect_ratio (Optional[str]): Requested output-image aspect ratio.
		size (Optional[str]): Requested output-image size.
		resolution (Optional[str]): Media-resolution value retained for UI compatibility.
		max_output_tokens (Optional[int]): Maximum text-output token count.
		output_mime_type (Optional[str]): Requested output-image MIME type.
		response_mode (Optional[str]): Requested response-modality selection.
		response_format_value (Optional[Any]): Interactions response-format configuration.
		generation_config (Dict[str, Any]): Interactions generation configuration.
		input_content (List[Dict[str, Any]]): Multimodal Interactions input.
		tool_objects (List[Dict[str, Any]]): Interactions Google Search tool declarations.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		response (Optional[Any]): Raw response used by application token accounting.
		output_text (str): Text extracted from the latest Interaction.
		output_image_content (Optional[Any]): Image content from the latest Interaction.
		grounding_metadata (Optional[Any]): Grounding data retained for UI compatibility.
		grounding_sources (List[Dict[str, Any]]): Normalized grounding records.
	"""
	
	client: Optional[ genai.Client ]
	http_options: HttpOptions
	aspect_ratio: Optional[ str ]
	size: Optional[ str ]
	resolution: Optional[ str ]
	max_output_tokens: Optional[ int ]
	output_mime_type: Optional[ str ]
	response_mode: Optional[ str ]
	response_format_value: Optional[ Any ]
	generation_config: Dict[ str, Any ]
	input_content: List[ Dict[ str, Any ] ]
	tool_objects: List[ Dict[ str, Any ] ]
	interaction: Optional[ Any ]
	response: Optional[ Any ]
	output_text: str
	output_image_content: Optional[ Any ]
	grounding_metadata: Optional[ Any ]
	grounding_sources: List[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gemini-3.1-flash-image' ) -> None:
		"""Initialize the Images wrapper.

		Purpose:
			Initializes image-model configuration, Interactions request state, output
			configuration, and response placeholders. The constructor performs local state
			assignment only and does not create a client or submit a provider request.

		Args:
			model (str): Default Gemini image model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.model = model
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.number = 1
		self.instructions = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_tokens = None
		self.max_output_tokens = None
		self.aspect_ratio = None
		self.size = None
		self.resolution = None
		self.media_resolution = None
		self.output_mime_type = None
		self.response_mode = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.tool_choice = None
		self.input_content = [ ]
		self.response_format_value = None
		self.generation_config = { }
		self.tool_objects = [ ]
		self.interaction = None
		self.response = None
		self.content_response = None
		self.image_response = None
		self.output_text = ''
		self.output_image_content = None
		self.grounding_metadata = None
		self.grounding_sources = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini image models.

		Purpose:
			Provides image-generation and image-editing model identifiers exposed by the Jeni
			Images mode.

		Returns:
			List[str]: Supported Gemini image-model identifiers.
		"""
		return [ 'gemini-3.1-flash-lite-image', 'gemini-3.1-flash-image', 'gemini-3-pro-image', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return compatibility include options.

		Purpose:
			Preserves the option contract consumed by existing Jeni controls.

		Returns:
			List[str]: Existing include-option values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content', ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""Return supported output-image aspect ratios.

		Purpose:
			Provides aspect-ratio values accepted by Gemini image response formats.

		Returns:
			List[str]: Supported aspect ratios.
		"""
		return [ '1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1',
			'9:16', '16:9', '21:9', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Purpose:
			Provides media-resolution values retained by the Jeni image controls.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported image response modes.

		Purpose:
			Provides text-only, image-only, and combined text-and-image output selections.

		Returns:
			List[str]: Supported response-mode values.
		"""
		return [ 'text', 'image', 'text_and_image', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported thinking levels.

		Purpose:
			Provides thinking-level values accepted by supported image models.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'unspecified', 'minimal', 'low', 'medium', 'high', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Return supported output-image sizes.

		Purpose:
			Provides output-image sizes accepted through the Interactions response format.

		Returns:
			List[str]: Supported image-size values.
		"""
		return [ '512', '1K', '2K', '4K', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported image grounding tools.

		Purpose:
			Provides Google Web Search and Google Image Search selections used by the Jeni
			Images mode.

		Returns:
			List[str]: Supported image tool identifiers.
		"""
		return [ 'google_search', 'image_search', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice values.

		Purpose:
			Preserves the existing Jeni tool-choice option contract.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'any', 'none', 'validated', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported text-output MIME types.

		Purpose:
			Provides text MIME types retained by the Jeni image controls.

		Returns:
			List[str]: Supported text-output MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Return supported generated-image MIME types.

		Purpose:
			Provides image MIME types accepted by the Interactions image response format.

		Returns:
			List[str]: Supported generated-image MIME types.
		"""
		return [ 'image/jpeg', 'image/png', 'image/webp', ]
	
	@property
	def resolution_options( self ) -> List[ str ]:
		"""Return supported output-image resolution options.

		Purpose:
			Provides the output-size values exposed by the Jeni Images mode.

		Returns:
			List[str]: Supported output-image sizes.
		"""
		return list( self.size_options )
	
	def supports_image_size( self, model: str ) -> bool:
		"""Return whether the selected model supports explicit image size.

		Purpose:
			Centralizes output-image-size feature gating for the Jeni image controls.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when the model supports explicit image size.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.image_size_models = { 'gemini-3.1-flash-lite-image', 'gemini-3.1-flash-image',
				'gemini-3-pro-image', }
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str ) -> bool:
		"""Return whether the model supports Google Search grounding.

		Purpose:
			Centralizes Google Search grounding feature gating for image generation.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when the model supports Google Search grounding.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.search_grounding_models = { 'gemini-3.1-flash-image', 'gemini-3-pro-image', }
			return self.model_name in self.search_grounding_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_search_grounding( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str ) -> bool:
		"""Return whether the model supports Google Image Search grounding.

		Purpose:
			Restricts Google Image Search grounding to the model documented for that feature.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when Google Image Search is supported.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
		image_only: bool = False ) -> List[ str ]:
		"""Normalize the requested response modes.

		Purpose:
			Converts the Jeni response-mode value into text and image modality identifiers used
			to build the Interactions response format.

		Args:
			response_modalities (Optional[str]): Jeni response-mode value.
			image_only (bool): Whether image output must be included.

		Returns:
			List[str]: Normalized response modes.

		Raises:
			Error: Raised when response-mode normalization fails.
		"""
		try:
			self.response_mode = response_modalities
			self.image_only = image_only
			self.mode_name = str( self.response_mode or '' ).strip( ).lower( )
			if self.mode_name == 'text_and_image':
				return [ 'text', 'image', ]
			
			if self.mode_name == 'text':
				return [ 'text', ]
			
			if self.mode_name == 'image':
				return [ 'image', ]
			
			if self.image_only:
				return [ 'image', ]
			
			return [ 'text', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'normalize_response_modalities( self, **kwargs ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def get_image_mime_type( self, path: str ) -> str:
		"""Return the MIME type for a local image.

		Purpose:
			Uses the local file suffix to produce the image MIME type required by an
			Interactions image-content block.

		Args:
			path (str): Local image path.

		Returns:
			str: Image MIME type.

		Raises:
			Error: Raised when validation or MIME-type resolution fails.
			ValueError: Raised when ``path`` is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_suffix = Path( self.file_path ).suffix.lower( )
			if self.file_suffix in ('.jpg', '.jpeg'):
				return 'image/jpeg'
			
			if self.file_suffix == '.webp':
				return 'image/webp'
			
			if self.file_suffix == '.gif':
				return 'image/gif'
			
			return 'image/png'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_image_mime_type( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_image_input( self, prompt: str,
		path: Optional[ str ]=None ) -> List[ Dict[ str, Any ] ]:
		"""Build Interactions text and image input.

		Purpose:
			Creates a text content block and, when supplied, a base64-encoded local image
			content block for image analysis or editing.

		Args:
			prompt (str): Image-generation, analysis, or editing instruction.
			path (Optional[str]): Optional local image path.

		Returns:
			List[Dict[str, Any]]: Interactions-compatible input content.

		Raises:
			Error: Raised when validation, file reading, or input construction fails.
			ValueError: Raised when ``prompt`` is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.file_path = path
			self.input_content = [ { 'type': 'text', 'text': self.prompt.strip( ), }, ]
			if self.file_path:
				self.image_bytes = Path( self.file_path ).read_bytes( )
				throw_if( 'image_bytes', self.image_bytes )
				self.image_data = base64.b64encode( self.image_bytes ).decode( 'utf-8' )
				self.image_mime_type = self.get_image_mime_type( self.file_path )
				self.input_content.append( { 'type': 'image', 'data': self.image_data,
					'mime_type': self.image_mime_type, } )
			
			return self.input_content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_image_input( self, **kwargs) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: Optional[ float ] = None,
		top_p: Optional[ float ] = None, max_tokens: Optional[ int ] = None ) -> Dict[ str, Any ]:
		"""Build the Interactions image generation configuration.

		Purpose:
			Converts supported Jeni inference values into the Interactions generation
			configuration.

		Args:
			temperature (Optional[float]): Sampling temperature.
			top_p (Optional[float]): Top-p sampling value.
			max_tokens (Optional[int]): Maximum output-token count.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when generation configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.generation_config = { }
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_output_tokens is not None and self.max_output_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = (self.max_output_tokens)
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_generation_config( self, **kwargs ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_modalities: Optional[ str ], image_only: bool=False,
		aspect: Optional[ str ] = None, resolution: Optional[ str ] = None,
		output_mime_type: Optional[ str ] = None ) -> Any:
		"""Build the Interactions image response format.

		Purpose:
			Constructs text, image, or combined response formats and applies supported image
			MIME type, aspect ratio, and output-size settings to the image format.

		Args:
			response_modalities (Optional[str]): Jeni response-mode value.
			image_only (bool): Whether image output must be included.
			aspect (Optional[str]): Requested output-image aspect ratio.
			resolution (Optional[str]): Requested output-image size.
			output_mime_type (Optional[str]): Requested generated-image MIME type.

		Returns:
			Any: Interactions response-format object or list.

		Raises:
			Error: Raised when response-format construction fails.
		"""
		try:
			self.response_mode = response_modalities
			self.image_only = image_only
			self.aspect_ratio = aspect
			self.size = resolution
			self.output_mime_type = output_mime_type
			self.response_modalities = self.normalize_response_modalities(
				response_modalities=self.response_mode, image_only=self.image_only )
			self.response_formats: List[ Dict[ str, Any ] ] = [ ]
			for modality in self.response_modalities:
				if modality == 'text':
					self.response_formats.append( { 'type': 'text', } )
				
				elif modality == 'image':
					self.image_format: Dict[ str, Any ] = { 'type': 'image', }
					if self.output_mime_type:
						self.image_format[ 'mime_type' ] = (self.output_mime_type)
					
					if self.aspect_ratio:
						self.image_format[ 'aspect_ratio' ] = (self.aspect_ratio)
					
					if (self.size and self.supports_image_size( self.model )):
						self.image_format[ 'image_size' ] = self.size
					
					self.response_formats.append( self.image_format )
			
			if len( self.response_formats ) == 1:
				self.response_format_value = self.response_formats[ 0 ]
			else:
				self.response_format_value = self.response_formats
			
			return self.response_format_value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_response_format( self, **kwargs ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tools( self, grounded: bool=False,
		image_search: bool=False ) -> List[ Dict[ str, Any ] ]:
		"""Build image grounding tool declarations.

		Purpose:
			Creates the Interactions Google Search tool declaration for web grounding and
			optionally enables Google Image Search when supported by the selected model.

		Args:
			grounded (bool): Whether Google Web Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			List[Dict[str, Any]]: Interactions tool declarations.

		Raises:
			Error: Raised when grounding-tool construction fails.
		"""
		try:
			self.grounded = grounded
			self.image_search = image_search
			self.tool_objects = [ ]
			if not self.grounded and not self.image_search:
				return self.tool_objects
			
			if not self.supports_search_grounding( self.model ):
				return self.tool_objects
			
			self.search_types: List[ str ] = [ ]
			if self.grounded:
				self.search_types.append( 'web_search' )
			
			if (self.image_search and self.supports_image_search( self.model )):
				self.search_types.append( 'image_search' )
			
			self.search_tool: Dict[ str, Any ] = { 'type': 'google_search', }
			
			if self.search_types:
				self.search_tool[ 'search_types' ] = self.search_types
			
			self.tool_objects.append( self.search_tool )
			return self.tool_objects
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_grounding_tools( self, **kwargs ) -> List[Dict[str, Any]]'
			Logger( ).write( exception )
			raise exception
	
	def extract_image( self, interaction: Any ) -> Optional[ PIL.Image.Image ]:
		"""Extract a generated image from an Interaction.

		Purpose:
			Reads the SDK ``output_image`` convenience property, decodes its base64 content,
			and returns an independent Pillow image object.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when no image was returned.

		Raises:
			Error: Raised when validation or image extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.output_image_content = getattr( self.interaction, 'output_image', None )
			if self.output_image_content is None:
				return None
			
			self.output_image_data = getattr( self.output_image_content, 'data', None )
			if not self.output_image_data:
				return None
			
			self.decoded_image = base64.b64decode( self.output_image_data )
			with PIL.Image.open( io.BytesIO( self.decoded_image ) ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'extract_image( self, interaction: Any ) -> Optional[ Image ]'
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, interaction: Any ) -> Optional[ str ]:
		"""Extract generated text from an Interaction.

		Purpose:
			Reads the Interactions SDK ``output_text`` convenience property and returns the
			normalized text expected by the Jeni image-analysis workflow.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			Optional[str]: Generated text, or None when no text was returned.

		Raises:
			Error: Raised when validation or text extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			return self.output_text or None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('extract_text( self, interaction: Any ) -> Optional[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture grounding information from the latest Interaction.

		Purpose:
			Retains the latest Interaction steps and Google Search result metadata for
			application display and diagnostics.

		Returns:
			None: This method updates response state through side effects.

		Raises:
			Error: Raised when grounding metadata cannot be captured.
		"""
		try:
			self.grounding_metadata = None
			self.grounding_sources = [ ]
			
			if self.interaction is None:
				return
			
			self.steps = getattr( self.interaction, 'steps', None ) or [ ]
			for step in self.steps:
				self.step_type = str( getattr( step, 'type', '' ) or '' ).strip( )
				if self.step_type != 'google_search_result':
					continue
				
				self.result = getattr( step, 'result', None )
				self.grounding_sources.append( { 'type': self.step_type, 'result': self.result, } )
			
			if self.grounding_sources:
				self.grounding_metadata = self.grounding_sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'capture_metadata( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def execute_interaction( self, prompt: str, model: str, path: Optional[ str ] = None,
		aspect: Optional[ str ] = None, number: Optional[ int ] = None,
		temperature: Optional[ float ] = None, top_p: Optional[ float ] = None,
		frequency: Optional[ float ] = None, presence: Optional[ float ] = None,
		max_tokens: Optional[ int ] = None, resolution: Optional[ str ] = None,
		instruct: Optional[ str ] = None, output_mime_type: Optional[ str ] = None,
		response_modalities: Optional[ str ] = None, image_only: bool = False,
		grounded: bool = False, image_search: bool = False ) -> Any:
		"""Execute an image Interaction.

		Purpose:
			Validates required inputs, creates the Gemini client, builds multimodal input,
			generation settings, output formats, and grounding tools, submits the request, and
			captures the response state shared by generation, analysis, and editing workflows.

		Args:
			prompt (str): Image-generation, analysis, or editing instruction.
			model (str): Gemini image-model identifier.
			path (Optional[str]): Optional local image path.
			aspect (Optional[str]): Requested output-image aspect ratio.
			number (Optional[int]): Requested result count retained for compatibility.
			temperature (Optional[float]): Sampling temperature.
			top_p (Optional[float]): Top-p sampling value.
			frequency (Optional[float]): Frequency penalty retained for compatibility.
			presence (Optional[float]): Presence penalty retained for compatibility.
			max_tokens (Optional[int]): Maximum output-token count.
			resolution (Optional[str]): Requested generated-image size.
			instruct (Optional[str]): System instruction text.
			output_mime_type (Optional[str]): Requested generated-image MIME type.
			response_modalities (Optional[str]): Requested response-mode value.
			image_only (bool): Whether image output must be included.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Any: Completed Gemini Interaction.

		Raises:
			Error: Raised when validation, request construction, or provider execution fails.
			ValueError: Raised when ``prompt``, ``model``, or the API key is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.file_path = path
			self.aspect_ratio = aspect
			self.number = number
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.size = resolution
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.image_only = image_only
			self.grounded = grounded
			self.image_search = image_search
			self.api_key = self.gemini_api_key or self.google_api_key
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			self.input_content = self.build_image_input( prompt=self.prompt, path=self.file_path )
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_output_tokens )
			
			self.response_format_value = self.build_response_format(
				response_modalities=self.response_mode, image_only=self.image_only,
				aspect=self.aspect_ratio, resolution=self.size,
				output_mime_type=self.output_mime_type )
			
			self.tool_objects = self.build_grounding_tools( grounded=self.grounded,
				image_search=self.image_search )
			
			self.request: Dict[ str, Any ] = { 'model': self.model, 'input': self.input_content,
				'response_format': self.response_format_value, 'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.image_response = self.interaction
			self.capture_metadata( )
			return self.interaction
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'execute_interaction( self, **kwargs ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Return the image from the most recent Interaction.

		Purpose:
			Preserves the existing Jeni helper contract by extracting the generated image from
			the latest Interaction.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when unavailable.

		Raises:
			Error: Raised when image extraction fails.
		"""
		try:
			if self.interaction is None:
				return None
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_first_image( self ) -> Optional[ PIL.Image.Image ]')
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Return text from the most recent Interaction.

		Purpose:
			Preserves the existing Jeni helper contract by extracting generated text from the
			latest Interaction.

		Returns:
			Optional[str]: Generated text, or None when unavailable.

		Raises:
			Error: Raised when text extraction fails.
		"""
		try:
			if self.interaction is None:
				return None
			
			return self.extract_text( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_output_text( self ) -> Optional[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-3.1-flash-image', aspect: str = None,
		number: int=None, temperature: float=None, top_p: float=None, frequency: float=None,
		presence: float = None, max_tokens: int = None, resolution: str = None,
		instruct: str = None, output_mime_type: str = None, response_modalities: str = None,
		grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Generate an image through the Gemini Interactions API.

		Purpose:
			Submits a text-to-image Interaction and returns the generated Pillow image expected
			by the Jeni Image Generation workflow.

		Args:
			prompt (str): Image-generation instruction.
			model (str): Gemini image-model identifier.
			aspect (str): Requested output-image aspect ratio.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Requested generated-image size.
			instruct (str): System instruction text.
			output_mime_type (str): Requested generated-image MIME type.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when no image is returned.

		Raises:
			Error: Raised when request execution or image extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			self.interaction = self.execute_interaction( prompt=prompt, model=model, path=None,
				aspect=aspect, number=number, temperature=temperature, top_p=top_p,
				frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities, image_only=True, grounded=grounded,
				image_search=image_search )
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, **kwargs ) -> Optional[ PIL.Image.Image ]'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-3.1-flash-image',
		aspect: str = None, number: int = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		resolution: str = None, instruct: str = None, output_mime_type: str = None,
		response_modalities: str=None, grounded: bool=False,
		image_search: bool=False ) ->  Optional[ str ]:
		"""Analyze an image through the Gemini Interactions API.

		Purpose:
			Submits text and a local image as multimodal Interaction input and returns the
			generated text expected by the Jeni Image Analysis workflow.

		Args:
			prompt (str): Image-analysis instruction.
			path (str): Local image path.
			model (str): Gemini image-model identifier.
			aspect (str): Aspect-ratio value retained for compatibility.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Media-resolution value retained for compatibility.
			instruct (str): System instruction text.
			output_mime_type (str): Output MIME type retained for compatibility.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[str]: Generated image analysis, or None when no text is returned.

		Raises:
			Error: Raised when request execution or text extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.interaction = self.execute_interaction( prompt=prompt, model=model,
				path=self.file_path, aspect=aspect, number=number, temperature=temperature,
				top_p=top_p, frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities or 'text', image_only=False,
				grounded=grounded, image_search=image_search )
			
			return self.extract_text( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, **kwargs ) -> Optional[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-3.1-flash-image',
		aspect: str = None, number: int = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		resolution: str = None, instruct: str = None, output_mime_type: str = None,
		response_modalities: str = None, grounded: bool = False,
		image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Edit an image through the Gemini Interactions API.

		Purpose:
			Submits an editing instruction and local image as multimodal Interaction input and
			returns the generated Pillow image expected by the Jeni Image Editing workflow.

		Args:
			prompt (str): Image-editing instruction.
			path (str): Local source-image path.
			model (str): Gemini image-model identifier.
			aspect (str): Requested output-image aspect ratio.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Requested generated-image size.
			instruct (str): System instruction text.
			output_mime_type (str): Requested generated-image MIME type.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[PIL.Image.Image]: Edited image, or None when no image is returned.

		Raises:
			Error: Raised when request execution or image extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.interaction = self.execute_interaction( prompt=prompt, model=model,
				path=self.file_path, aspect=aspect, number=number, temperature=temperature,
				top_p=top_p, frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities or 'image', image_only=True,
				grounded=grounded, image_search=image_search )
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, **kwargs ) -> Optional[ PIL.Image.Image ]'
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
	
	def __init__( self, model: str = 'gemini-embedding-001' ):
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
	
	def build_embedding_config( self, model: str = 'gemini-embedding-001', dimensions: int = None,
		task_type: str = None, title: str = None ) -> EmbedContentConfig:
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
			
			if (
					self.title and self.task_type == 'RETRIEVAL_DOCUMENT' and 'gemini-embedding-2'
					not in self.model):
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
	
	def create( self, text: str | List[ str ], model: str = 'gemini-embedding-001',
		dimensions: int = None, task_type: str = None, title: str = None,
		encoding_format: str = 'float' ) -> List[ float ] | List[ List[ float ] ] | None:
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
	
	def __init__( self, model: str = 'gemini-2.5-flash-preview-tts' ):
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
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
		sample_width: int = 2 ) -> bytes:
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
					self.prompt_parts.append(
						'Read the following text at a faster, energetic pace.' )
			
			self.prompt_parts.append( str( text ).strip( ) )
			return '\n\n'.join( self.prompt_parts )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_tts_prompt( self, **kwargs ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, filepath: str = None,
		model: str = 'gemini-3.1-flash-tts-preview', format: str = 'audio/wav', speed: float =
		None,
		voice: str = None, frequency: float = None, presense: float = None, max_tokens: int = None,
		instruct: str = None, temperature: float = None,
		top_p: float = None ) -> bytes | str | None:
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
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float =
	0.8,
		top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000,
		instruct: str = None ):
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
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
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
	
	def build_prompt( self, language: str = None, start_time: float = None,
		end_time: float = None ) -> str:
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
		
		if (language is not None and str( language ).strip( ) and str(
				language ).strip( ) != 'Auto'):
			self.prompt_parts.append(
				f'The expected spoken language is {str( language ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only transcribe the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the transcript text.' )
		return ' '.join( self.prompt_parts )
	
	def transcribe( self, path: str, model: str = 'gemini-3-flash-preview', language: str = None,
		mime_type: str = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		start_time: float = None, end_time: float = None, instruct: str = None ) -> Optional[
		str ]:
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
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float =
	0.8,
		top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000,
		instruct: str = None ):
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
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
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
	
	def build_prompt( self, target: str, source: str = 'Auto', start_time: float = None,
		end_time: float = None ) -> str:
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
	
	def translate( self, path: str, model: str = 'gemini-3-flash-preview',
		language: str = 'English', source: str = 'Auto', mime_type: str = None,
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, start_time: float = None,
		end_time: float = None, instruct: str = None ) -> Optional[ str ]:
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
	
	def __init__( self, model: str = 'gemini-2.0-flash' ):
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
	
	def upload( self, filepath: str, name: str = None ) -> File | None:
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
	
	def list( self, model: str = 'gemini-3.0-flash', top_p: float = 0.8, top_k: int = 50,
		temperature: float = 0.5, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 8192, tool_choice: str = 'auto', stops: List[ str ] = None,
		tools: List[ str ] = None, domains: List[ str ] = None, modalities: List[ str ] = None,
		media_resolution: str = 'media_resolution_medium' ) -> Any | None:
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
	
	def summarize( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
	
	def search( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str = 'gemini-2.0-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None ) -> str | None:
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
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
	
	def delete( self, store_id: str, force: bool = True ) -> bool | Any:
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
	
	def upload( self, path: str, bucket: str, name: str = None ) -> Blob | None:
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
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str | None:
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
