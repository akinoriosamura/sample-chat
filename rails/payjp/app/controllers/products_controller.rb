class ProductsController < ApplicationController
    def index
    end

    def pay
        Payjp.api_key = 'sk_test_64931749fa88f77d12c184cc'
        charge = Payjp::Charge.create(
            :amount => 3500,
            :card => params['payjp-token'],
            :currency => 'jpy',
    )
    end
end
