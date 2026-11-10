
import re
from datetime import date

from presidio_analyzer import Pattern, PatternRecognizer


class NorwegianFodselsnummerRecognizer(PatternRecognizer):
    """
    Detect Norwegian fødselsnummer (11-digit identifiers).

    Validates:
    - 11-digit format
    - Calendar date
    - First control digit
    - Second control digit
    """

    def __init__(self):

        patterns = [
            Pattern(
                name="norwegian_fodselsnummer",
                regex=r"(?<!\d)\d{11}(?!\d)",
                score=0.85
            )
        ]

        super().__init__(
            supported_entity="NORWEGIAN_FODSELSNUMMER",
            patterns=patterns,
            context=[
                "fødselsnummer",
                "national identity number",
                "identity number",
                "personnummer",
                "national identification"
            ]
        )

    @staticmethod
    def calculate_control_digit(digits, weights):

        total = sum(
            int(digit) * weight
            for digit, weight in zip(digits, weights)
        )

        remainder = total % 11

        if remainder == 0:
            return 0

        if remainder == 1:
            return None

        return 11 - remainder

    @classmethod
    def validate_fodselsnummer(cls, number):

        if not re.fullmatch(r"\d{11}", number):
            return False

        day = int(number[0:2])
        month = int(number[2:4])
        year = int(number[4:6])

        # Validate calendar date.
        # The encoded century is not determined here.
        try:
            date(2000 + year, month, day)
        except ValueError:
            return False

        # Calculate first control digit.
        k1_weights = [3, 7, 6, 1, 8, 9, 4, 5, 2]

        k1 = cls.calculate_control_digit(
            number[:9],
            k1_weights
        )

        if k1 is None or k1 != int(number[9]):
            return False

        # Calculate second control digit.
        k2_weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

        k2 = cls.calculate_control_digit(
            number[:10],
            k2_weights
        )

        if k2 is None or k2 != int(number[10]):
            return False

        return True

    def validate_result(self, pattern_text):

        return self.validate_fodselsnummer(pattern_text)
