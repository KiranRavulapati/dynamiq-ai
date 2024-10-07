from dynamiq.memory import Memory
from dynamiq.memory.backend import InMemory
from dynamiq.prompts import MessageRole

# Create a memory instance with InMemory backend
memory = Memory(backend=InMemory())

# Add messages with metadata
memory.add_message(MessageRole.USER, "My favorite color is blue.", metadata={"topic": "colors", "user_id": "123"})
memory.add_message(MessageRole.ASSISTANT, "Blue is a calming color.", metadata={"topic": "colors", "user_id": "123"})
memory.add_message(MessageRole.USER, "I like red too.", metadata={"topic": "colors", "user_id": "456"})
memory.add_message(MessageRole.ASSISTANT, "Red is a passionate color.", metadata={"topic": "colors", "user_id": "456"})

# Search with filters only
results = memory.search_messages(filters={"user_id": "123"})
print("Results with filter only:", [r.content for r in results])

# Search with query and filters
results = memory.search_messages(query="color", filters={"user_id": "123"})
print("Results with query and filter:", [r.content for r in results])

# Search with query only
results = memory.search_messages("red")
print("Results with query only:", [r.content for r in results])

# Get all messages
messages = memory.get_all_messages()
print("All messages:")
for msg in messages:
    print(f"{msg.role}: {msg.content}")


print("Is memory empty?", memory.is_memory_empty())
