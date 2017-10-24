App.chat = App.cable.subscriptions.create("ChatChannel", {
  connected: function() {
    // Called when the subscription is ready for use on the server
  },

  disconnected: function() {
    // Called when the subscription has been terminated by the server
  },

  received: function(data) {
    return $('#messages').append(data['message']);
    // Called when there's incoming data on the websocket for this channel
  },

  post: function(message) {
    Message.create! content: data['message']
  }
}, $(document).on('keypress', '[data-behavior~=chat_post]', function(event) {
  if (event.keyCode === 13) {
    var chatForm = $('#chat-form');
    App.chat.post(chatForm.val());
    return chatForm.val('');
  }
}));

