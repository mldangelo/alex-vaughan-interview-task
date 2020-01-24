# Schema

Field Name | Description
-----------|-------------
artist_name | Represents the artist name.
artist_nationality | Represents the artist's country of origin.
artist_birth_year | Represents the artist's year of birth.
artist_death_year | Represents the artist's year of death. `-1` if still living.
auction_house | Name of auction house.
auction_sale_id | A unique id associated with the auction given by the auction house (ex. `PF1846`).
auction_department | The type of department an auction belongs to. Note: These are synthetic departments. We map auction house departments across auction houses into a common taxonomy (ex. `Post-War & Contemporary`, `-1` if unknown).
auction_location | Represents the city where the auction takes place.
auction_date | Represents the datetime when the auction started. Auction houses may report the date and time of the auction or only the date (ex. `2017-03-01 00:00:00` or `2016-11-18 15:00:00.489`. Ignore trailing milliseconds).
auction_currency | ISO currency code representing denomination used in the auction (ex. `USD`).
exchange_rate_to_usd | Exchange rate from original currency to USD on auction date.
auction_lot_count | Represents the number of lots in the auction (ex. `65`, `-1` if unknown).
lot_id | Unique lot id given by the auction house. These are alphanumeric (ex. `601` or `32T`)
lot_place_in_auction | The order in which the lot was offered during the auction.
lot_description | A json, html, or natural language description of the lot from the auction house website. These have not been cleaned by us.
lot_link | Link to auction lot listing. These are occasionally null.
work_title | Title of the work. These are occasionally messy and may contain dates. Many works are untitled.
work_medium | Medium of the work, such as `painting`, `drawing`, or `photograph`.
work_execution_year | Year when the work was is completed (ex. `1982`, `-1` if unknown).
work_dimensions | Unstructured text representing work dimensions (ex. `11 1/4 x 11 1/4 in. (28.6 x 28.6 cm.)`, `-1` if unknown).
work_height | Approximate height of the lot (`-1` if unknown).
work_width | Approximate width of the lot (`-1` if unknown).
work_depth | Approximate depth of the lot (`-1` if 2D or unknown).
unit | unit the measurements are in (ex. `cm`, `-1` if unknown).
hammer_price | The price at which a lot sold for at auction. If the lot was not sold, `-1` is recorded.
buyers_premium | The fee the auction house charges the buyer on top of the hammer price auction. Note that this is calculated after the lot is sold and should not be a feature of your model. `0` if the lot is unsold.
estimate_low | A the low estimate provided by the auction house prior to auction.
estimate_high | A high estimate hammer price provided by the auction house prior to auction.
