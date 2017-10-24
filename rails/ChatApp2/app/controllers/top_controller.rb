class TopController < ApplicationController
    def show
        @messages = Message.all
    end
end
