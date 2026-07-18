'''
  ******************************************************************************************
      Assembly:                Jeni
      Filename:                models.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="models.py" company="Terry D. Eppler">

	     Jeni is a df analysis tool integrating various Generative GPT, GptText-Processing, and
	     Machine-Learning algorithms for federal analysts.
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    models.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Message( ):
	'''

		Purpose:
		--------
		Represents a structured “Message” or instruction bundle used to interact with an LLM.
		This model is intended to capture the canonical components you pass into Jeni when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		content: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		role Optional[ str ]
			Optional a role associated with the content.

	'''
	role: Optional[ str ]
	content: Optional[ str ]

	def __init__( self, content: str, role: str ):
		self.role = role
		self.content = content

	def __dir__( self ) -> List[ str ]:
		return [ 'role', 'content' ]
	
class System( Message ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Jeni when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		content: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		role: Optional[ str ]
			Optional role associated with the message.

	'''
	
	def __init__(self, content: str, role: str='system' ):
		super( ).__init__( content, role )

	def __dir__( self ) -> List[ str ]:
		return [ 'role', 'content' ]
	

class User( Message ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Jeni when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		content: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		role: Optional[ str ]
			Optional role associated with the message.
	'''
	
	def __init__(self, content: str, role: str='user' ):
		super( ).__init__( content, role )

	def __dir__( self ) -> List[ str ]:
		return [ 'role', 'content' ]
	

class Prompt(  ):
	'''

		Purpose:
		--------
		Represents a structured “system prompt” or instruction bundle used to steer an LLM call.
		This model is intended to capture the canonical components you pass into Jeni when you
		want to track prompts as first-class objects (versioning, variables, and provenance).

		Attributes:
		----------
		content: Optional[ str ]
			Optional background context provided to the model (policies, references, etc.).

		role: Optional[ str ]
			Optional role associated with the message.
	'''
	system: Optional[ System ] = None
	user: Optional[ User ] = None
	
	def __init__( self, sys: System, user: User ):
		self.system = sys
		self.user = user

	def __dir__( self ) -> List[ str ]:
		return [ 'role', 'content' ]
	
	def build( self ) -> Dict[ str, str ]:
		'''

		Purpose:
		--------
		Method to creates a system/user prompt from Message objects

		Return:
			str - Optional background context provided to the model (policies, references, etc.).
		
		'''
		
class Error( BaseModel ):
	'''

		Purpose:
		--------
		Represents an error object returned by an upstream API. Jeni stores errors in structured
		form so UI and logging layers can surface details without brittle string parsing.

		Attributes:
		----------
		code: Optional[ str ]
			A short error code when available (e.g., "invalid_request_error").

		message: Optional[ str ]
			Human-readable error message.

	'''
	code: Optional[ str ] = None
	message: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class JsonSchema( BaseModel ):
	'''

		Purpose:
		--------
		Represents a JSON Schema-based response format definition. This is most commonly used
		when you request “structured outputs” constrained by a JSON schema.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the response format (commonly "json_schema").

		name: Optional[ str ]
			A friendly name for the schema/format.

		description: Optional[ str ]
			A description of the schema’s intent.

		schema: Optional[ Dict[ str, Any ] ]
			The JSON Schema definition object (draft-07 compatible patterns are common).

		strict: Optional[ bool ]
			Whether the upstream should enforce strict compliance with the schema.

	'''
	type: Optional[ str ] = None
	name: Optional[ str ] = None
	description: Optional[ str ] = None
	schema: Optional[ Dict[ str, Any ] ] = None
	strict: Optional[ bool ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class JsonObject( BaseModel ):
	'''

		Purpose:
		--------
		Represents a “JSON object” response format definition. This is typically used when you
		want JSON output without providing a full JSON Schema.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the response format (commonly "json_object").

	'''
	type: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Reasoning( BaseModel ):
	'''

		Purpose:
		--------
		Represents reasoning configuration and/or summary data surfaced by an upstream model.
		Jeni keeps this structured so callers can persist “reasoning metadata” without coupling
		to a specific vendor response shape.

		Attributes:
		----------
		effort: Optional[ str ]
			A label describing reasoning effort (when supported), e.g., "low", "medium", "high".

		summary: Optional[ str ]
			A short summary of reasoning (when the upstream provides it).

	'''
	effort: Optional[ str ] = None
	summary: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Math( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured “math solution” style payload (steps + final answer). Jeni uses
		this for workflows where a tool or model returns stepwise math output.

		Attributes:
		----------
		steps: Optional[ List[ Step ] ]
			Ordered list of intermediate steps with explanations and outputs.

		final_answer: Optional[ str ]
			Final answer string.

	'''
	class Step( BaseModel ):
		'''

			Purpose:
			--------
			Represents an individual intermediate computation or explanation step.

			Attributes:
			----------
			explanation: Optional[ str ]
				Natural language explanation of the step.

			output: Optional[ str ]
				The computed output (may be numeric or symbolic).

		'''
		explanation: Optional[ str ] = None
		output: Optional[ str ] = None

		class Config:
			arbitrary_types_allowed = True
			extra = 'ignore'

	steps: Optional[ List[ Step ] ] = None
	final_answer: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Document( BaseModel ):
	'''

		Purpose:
		--------
		Represents a generic “document-like” structured output. Jeni uses this for demos and
		for workflows where the model produces a multi-field narrative artifact with concepts.

		Attributes:
		----------
		invented_year: Optional[ int ]
			Example field used in some structured-output prompts; can be repurposed.

		summary: Optional[ str ]
			High-level summary of the document.

		inventors: Optional[ List[ str ] ]
			Example list field used in structured-output prompts.

		description: Optional[ str ]
			Long-form description.

		concepts: Optional[ List[ Concept ] ]
			List of extracted or described concepts.

	'''
	invented_year: Optional[ int ] = None
	summary: Optional[ str ] = None
	inventors: Optional[ List[ str ] ] = None
	description: Optional[ str ] = None

	class Concept( BaseModel ):
		'''

			Purpose:
			--------
			Represents a named concept extracted from or associated with a document.

			Attributes:
			----------
			title: Optional[ str ]
				Concept title/name.

			description: Optional[ str ]
				Concept description.

		'''
		title: Optional[ str ] = None
		description: Optional[ str ] = None

		class Config:
			arbitrary_types_allowed = True
			extra = 'ignore'

	concepts: Optional[ List[ Concept ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ResponseFormat( BaseModel ):
	'''

		Purpose:
		--------
		Represents a response formatting directive. This mirrors common “response_format”
		shapes where you can request plain text, JSON object output, or JSON schema-constrained
		structured output.

		Attributes:
		----------
		text: Optional[ Text ]
			Text response format information, when applicable.

		json_schema: Optional[ JsonSchema ]
			Schema definition for structured output.

		json_object: Optional[ JsonObject ]
			JSON object format directive.

	'''
	text: Optional[ str ] = None
	json_schema: Optional[ JsonSchema ] = None
	json_object: Optional[ JsonObject ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Function( BaseModel ):
	'''

		Purpose:
		--------
		Represents a tool/function descriptor for tool-calling. Jeni uses this to keep “tool
		specification” structured (name, description, JSON schema parameters, strictness).

		Attributes:
		----------
		name: Optional[ str ]
			Function name as exposed to the model.

		type: Optional[ str ]
			Type discriminator (commonly "function") when used in tool lists.

		description: Optional[ str ]
			Human-readable description of the tool/function.

		parameters: Optional[ Dict[ str, Any ] ]
			JSON Schema-like parameters object describing accepted inputs.

		strict: Optional[ bool ]
			Whether the upstream should strictly validate arguments against the schema.

	'''
	name: Optional[ str ] = None
	type: Optional[ str ] = None
	description: Optional[ str ] = None
	parameters: Optional[ Dict[ str, Any ] ] = None
	strict: Optional[ bool ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Message( BaseModel ):
	'''

		Purpose:
		--------
		Represents a chat message-like object used by Jeni to normalize conversational state.
		This is intentionally general to support both “input messages” and “output messages”.

		Attributes:
		----------
		content: str
			Message content payload. Jeni treats this as required for operational messages.

		role: str
			Message role (e.g., "system", "user", "assistant", "tool").

		type: Optional[ str ]
			Optional discriminator if an upstream system emits typed message objects.

		instructions: Optional[ str ]
			Optional per-message instruction string (used in some orchestration patterns).

		data: Optional[ Dict ]
			Optional message metadata or additional structured payload.

	'''
	content: str
	role: str
	type: Optional[ str ] = None
	instructions: Optional[ str ] = None
	data: Optional[ Dict ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Location( BaseModel ):
	'''

		Purpose:
		--------
		Represents a high-level user location descriptor used by web search or other tools.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for location objects.

		city: Optional[ str ]
			City name.

		country: Optional[ str ]
			Country name or code.

		region: Optional[ str ]
			State/province/region.

		timezone: Optional[ str ]
		IANA timezone string when known.

	'''
	type: Optional[ str ] = None
	city: Optional[ str ] = None
	country: Optional[ str ] = None
	region: Optional[ str ] = None
	timezone: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class GeoCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Represents a latitude/longitude coordinate pair, optionally with a timezone. This is
		useful for tools like web search, maps, or proximity-based retrieval.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for geocoordinate objects.

		latitude: Optional[ float ]
			Latitude in decimal degrees.

		longitude: Optional[ float ]
			Longitude in decimal degrees.

		timezone: Optional[ str ]
			IANA timezone string when known.

	'''
	type: Optional[ str ] = None
	latitude: Optional[ float ] = None
	longitude: Optional[ float ] = None
	timezone: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class FileSearch( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a file-search tool invocation. Jeni uses this to keep tool
		config structured and to support serialization/rehydration of tool configurations.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "file_search").

		vector_store_ids: Optional[ List[ str ] ]
			Vector store identifiers available to the search tool.

		max_num_results: Optional[ int ]
			Maximum number of results to return.

		filters: Optional[ Dict[ str, Any ] ]
			Optional filter object (metadata filters, namespace filters, etc.).

	'''
	type: Optional[ str ] = None
	vector_store_ids: Optional[ List[ str ] ] = None
	max_num_results: Optional[ int ] = None
	filters: Optional[ Dict[ str, Any ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class WebSearch( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a web-search tool invocation.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "web_search").

		search_context_size: Optional[ str ]
			Desired context size (vendor-specific; common values are "low", "medium", "high").

		user_location: Optional[ Any ]
			Optional location descriptor to bias search results. This may be a Location,
			GeoCoordinates, or a vendor-specific object.

	'''
	type: Optional[ str ] = None
	search_context_size: Optional[ str ] = None
	user_location: Optional[ Any ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ComputerUse( BaseModel ):
	'''

		Purpose:
		--------
		Represents configuration for a computer-use tool invocation (UI automation / virtual
		display sessions).

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the tool (commonly "computer_use").

		display_height: Optional[ int ]
			Height (pixels) of the virtual display.

		display_width: Optional[ int ]
			Width (pixels) of the virtual display.

		environment: Optional[ str ]
			Environment label (e.g., "browser", "desktop") when supported by the tool provider.

	'''
	type: Optional[ str ] = None
	display_height: Optional[ int ] = None
	display_width: Optional[ int ] = None
	environment: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class WxForecast( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified weather forecast payload returned by a tool or model.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		temperature: Optional[ int ]
			Temperature value (units depend on the tool/provider).

		precipitation: Optional[ int ]
			Precipitation percentage or amount (provider-specific).

		sky_conditions: Optional[ str ]
			Text description such as "clear", "cloudy", "rain".

	'''
	type: Optional[ str ] = None
	temperature: Optional[ int ] = None
	precipitation: Optional[ int ] = None
	sky_conditions: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class Directions( BaseModel ):
	'''

		Purpose:
		--------
		Represents a simplified directions/route payload returned by a mapping tool.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		route: Optional[ Any ]
			Route representation (provider-specific). Frequently this is a string/polyline or
			a structured list of steps.

	'''
	type: Optional[ str ] = None
	route: Optional[ Any ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class ValidationStatus( BaseModel ):
	'''

		Purpose:
		--------
		Represents validation results produced by Jeni (or by a tool/model) when generating
		structured artifacts (SQL, JSON, schemas, etc.).

		Attributes:
		----------
		is_valid: Optional[ bool ]
			True if the artifact passed validation; False otherwise.

		syntax_errors: Optional[ List[ str ] ]
			List of syntax/validation error messages. Empty or None when valid.

	'''
	is_valid: Optional[ bool ] = None
	syntax_errors: Optional[ List[ str ] ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class SqlStatement( BaseModel ):
	'''

		Purpose:
		--------
		Represents a canonical set of SQL commands associated with a table. This model is
		useful when a model/tool generates multiple related statements (DDL + DML patterns).

		Attributes:
		----------
		table_name: Optional[ str ]
			Name of the table associated with the commands.

		create_command: Optional[ str ]
			CREATE TABLE or equivalent DDL.

		select_command: Optional[ str ]
			SELECT statement template or example query.

		insert_command: Optional[ str ]
			INSERT statement template.

		delete_command: Optional[ str ]
			DELETE statement template.

		update_command: Optional[ str ]
			UPDATE statement template.

	'''
	table_name: Optional[ str ] = None
	create_command: Optional[ str ] = None
	select_command: Optional[ str ] = None
	insert_command: Optional[ str ] = None
	delete_command: Optional[ str ] = None
	update_command: Optional[ str ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class SQLQueryGeneration( BaseModel ):
	'''

		Purpose:
		--------
		Represents a structured SQL query generation result. This is intended for Jeni’s
		SQL-oriented tooling where you want the SQL plus metadata and validation.

		Attributes:
		----------
		query: str
			The generated SQL query.

		query_type: str
			Query type label (e.g., "SELECT", "INSERT", "UPDATE", "DDL").

		tables_used: List[ str ]
			List of table names referenced by the query.

		estimated_complexity: str
			Complexity label (e.g., "simple", "moderate", "complex").

		execution_notes: List[ str ]
			Notes about execution assumptions, indexes, filters, or constraints.

		validation_status: ValidationStatus
			Validation status object describing whether the SQL is syntactically/semantically valid.

	'''
	query: Optional[ str ]
	query_type: Optional[ str ]
	tables_used: Optional[ List[ str ] ]
	estimated_complexity: Optional[ str ]
	execution_notes: Optional[ List[ str ] ]
	validation_status: Optional[ ValidationStatus ]

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'


class SkyCoordinates( BaseModel ):
	'''

		Purpose:
		--------
		Represents right ascension / declination coordinate pairs used in astronomy-oriented
		structured outputs.

		Attributes:
		----------
		type: Optional[ str ]
			Type discriminator for the object.

		declination: Optional[ float ]
			Declination in decimal degrees.

		right_ascension: Optional[ float ]
			Right ascension in decimal degrees or hours (provider-specific).

	'''
	type: Optional[ str ] = None
	declination: Optional[ float ] = None
	right_ascension: Optional[ float ] = None

	class Config:
		arbitrary_types_allowed = True
		extra = 'ignore'
