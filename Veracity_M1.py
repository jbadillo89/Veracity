from typing import TextIO
from conversation.engine import ConversationEngine
from models.session import Session
def main(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> Session:
    return ConversationEngine(input_stream, output_stream).run()
if __name__ == "__main__":
    main()
