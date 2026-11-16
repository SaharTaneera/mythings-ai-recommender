// Example for an existing ASP.NET Core commerce backend; NOT a live MyThings integration.
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace MyThings.Recommendations;

public sealed record CartItemDto(
    [property: JsonPropertyName("product_id")] string ProductId,
    [property: JsonPropertyName("quantity")] int Quantity);

public sealed record CartDto(
    [property: JsonPropertyName("merchant_id")] string MerchantId,
    [property: JsonPropertyName("items")] IReadOnlyList<CartItemDto> Items);

public sealed record ProductDto(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("merchant_id")] string MerchantId,
    [property: JsonPropertyName("name_en")] string? NameEn,
    [property: JsonPropertyName("name_ar")] string? NameAr,
    [property: JsonPropertyName("category")] string? Category = null,
    [property: JsonPropertyName("subcategory")] string? Subcategory = null,
    [property: JsonPropertyName("description_en")] string? DescriptionEn = null,
    [property: JsonPropertyName("description_ar")] string? DescriptionAr = null);

public sealed record RecommendationRequestDto(
    [property: JsonPropertyName("cart")] CartDto Cart,
    [property: JsonPropertyName("products")] IReadOnlyList<ProductDto> Products,
    [property: JsonPropertyName("candidates")] IReadOnlyList<ProductDto> Candidates,
    [property: JsonPropertyName("top_k")] int TopK = 3);

public sealed record RecommendedProductDto(
    [property: JsonPropertyName("product_id")] string ProductId,
    [property: JsonPropertyName("score")] double Score);

public sealed record RecommendationResponseDto(
    [property: JsonPropertyName("recommendations")] IReadOnlyList<RecommendedProductDto> Recommendations,
    [property: JsonPropertyName("model_version")] string ModelVersion);

public sealed class MyThingsRecommendationClient
{
    private readonly HttpClient _client;

    public MyThingsRecommendationClient(HttpClient client)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }

    public async Task<RecommendationResponseDto> RecommendAsync(
        RecommendationRequestDto request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        // The host backend must authorize the cart and select a bounded, in-stock,
        // same-merchant candidate list before calling this demonstration service.
        using var response = await _client.PostAsJsonAsync(
            "api/v1/recommendations", request, cancellationToken);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync<RecommendationResponseDto>(
            cancellationToken: cancellationToken
        ) ?? throw new InvalidOperationException("Empty recommendation response.");
    }
}
