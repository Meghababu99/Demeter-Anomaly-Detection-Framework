class Hex(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_json_schema__(cls, schema,handler):
        schema.update(
            pattern='^[0-9A-Fa-f]{1,9}$',
            examples=['13ff', 'c56fds'],
        )
        return schema

    @classmethod
    def validate(cls, v, info=None):
        if not isinstance(v, str):
            raise TypeError('string required')

        try:
            vv = hex(int(v, 16))
        except ValueError:
            raise ValueError('invalid hex format')

        return cls(vv)

    def __repr__(self):
        return f'Hex({super().__repr__()})'


# class Bool(str):
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def __get_pydantic_json_schema__(cls, schema):
#         schema.update(
#             pattern='^[0,1]{1}$',
#             examples=['0', '1'],
#         )
#         return schema

#     @classmethod
#     def validate(cls, v):
#         if not isinstance(v, str, info=None):
#             raise TypeError('string required')

#         try:
#             vv = bool(int(v))
#         except ValueError:
#             raise ValueError('invalid bool format')

#         return cls(vv)

#     def __repr__(self):
#         return f'Bool({super().__repr__()})'
class Bool(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        schema.update(
            pattern='^[0,1]{1}$',
            examples=['0', '1'],
        )
        return schema

    @classmethod
    def validate(cls, v, info=None):  # Add 'info' as an optional parameter
        if not isinstance(v, str):
            raise TypeError('string required')

        try:
            vv = bool(int(v))
        except ValueError:
            raise ValueError('invalid bool format')

        return cls(vv)

    def __repr__(self):
        return f'Bool({super().__repr__()})'
